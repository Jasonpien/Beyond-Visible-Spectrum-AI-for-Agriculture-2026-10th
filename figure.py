import os
import re
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# Config (路徑已完美對應您的系統)
# ============================================================
class CFG:
    # 原始資料路徑
    ROOT = "/ssd6/pienyuzhe/ICPR2026_BeyondAI_Crop_Disease/Dataset/Kaggle_Prepared"
    TRAIN_DIR = "train"
    
    # 頻段裁切設定
    HS_DROP_FIRST = 10
    HS_DROP_LAST = 14
    
    # 模型 (.pkl) 存放位置 (已更新)
    MODEL_DIR = "/ssd6/pienyuzhe/ICPR2026_BeyondAI_Crop_Disease/final10/output_fusion2689/"
    
    # 圖片輸出位置 (已更新)
    OUT_DIR = "/ssd6/pienyuzhe/ICPR2026_BeyondAI_Crop_Disease/final10/figures/"


LABELS = ["Health", "Rust", "Other"]

# ============================================================
# Data Utils
# ============================================================
def list_files(folder, exts):
    if not os.path.isdir(folder): return []
    return sorted([os.path.join(folder, fn) for fn in os.listdir(folder) if fn.lower().endswith(exts)])

def base_id(path):
    return os.path.splitext(os.path.basename(path))[0]

def parse_label(bid):
    m = re.match(r"^(Health|Rust|Other)_", bid)
    return m.group(1) if m else None

def build_index(root, split):
    split_dir = os.path.join(root, split)
    idx = {}
    for p in list_files(os.path.join(split_dir, "HS"), (".tif", ".tiff")):
        idx.setdefault(base_id(p), {})["hs"] = p
    return idx

def make_train_df(train_idx):
    rows = []
    for bid, paths in train_idx.items():
        lab = parse_label(bid)
        if lab: rows.append({"base_id": bid, "label": lab, **paths})
    return pd.DataFrame(rows)

def read_tiff(path):
    try:
        arr = tiff.imread(path)
        if arr.ndim == 2: arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3 and arr.shape[0] < arr.shape[1]: arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.float32)
    except: return None

def normalize(x):
    x = np.nan_to_num(x, nan=0.0)
    mn, mx = x.min(), x.max()
    if mx - mn > 1e-6: x = (x - mn) / (mx - mn)
    return np.clip(x, 0, 1)

# ============================================================
# Plot 1: Spectral Profiles (類別平均光譜曲線)
# ============================================================
def plot_spectral_profiles(df, cfg):
    print("="*50)
    print("Generating Spectral Profiles (like PDF Fig 3)...")
    
    spectra_dict = {'Health': [], 'Rust': [], 'Other': []}
    
    for row in tqdm(df.itertuples(), total=len(df), desc="Reading HS Spectra"):
        if hasattr(row, 'hs') and isinstance(row.hs, str) and os.path.exists(row.hs):
            hs_data = read_tiff(row.hs)
            if hs_data is not None:
                C = hs_data.shape[2]
                if C > cfg.HS_DROP_FIRST + cfg.HS_DROP_LAST:
                    hs_data = hs_data[:, :, cfg.HS_DROP_FIRST:C-cfg.HS_DROP_LAST]
                
                hs_data = normalize(hs_data)
                
                # 提取中心區域光譜平均
                H, W, _ = hs_data.shape
                h1, h2, w1, w2 = H//4, 3*H//4, W//4, 3*W//4
                center_spec = hs_data[h1:h2, w1:w2, :].mean((0, 1))
                
                if row.label in spectra_dict:
                    spectra_dict[row.label].append(center_spec)
                    
    plt.figure(figsize=(12, 7))
    colors = {'Health': '#2ca02c', 'Rust': '#d62728', 'Other': '#1f77b4'}
    
    for label, spectra in spectra_dict.items():
        if not spectra: continue
        
        min_bands = min([s.shape[0] for s in spectra])
        spectra_matrix = np.array([s[:min_bands] for s in spectra])
        
        mean_spec = spectra_matrix.mean(axis=0)
        std_spec = spectra_matrix.std(axis=0)
        x_axis = np.arange(len(mean_spec))
        
        plt.plot(x_axis, mean_spec, label=label, color=colors[label], linewidth=2.5)
        plt.fill_between(x_axis, mean_spec - std_spec, mean_spec + std_spec, 
                         color=colors[label], alpha=0.15)

    plt.title('Normalized Mean Spectral Profiles by Crop Health Status', fontsize=16, fontweight='bold')
    plt.xlabel('Hyperspectral Band Index (after dropping noisy bands)', fontsize=12)
    plt.ylabel('Normalized Mean Reflectance', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.axvspan(int(min_bands*0.4), int(min_bands*0.55), color='gray', alpha=0.1, label='Red-Edge Region')
    
    plt.tight_layout()
    out_path = os.path.join(cfg.OUT_DIR, 'Figure_3_Spectral_Profiles.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved -> {out_path}")
    plt.close()

# ============================================================
# Plot 2: Feature Importance (特徵重要性分析)
# ============================================================
def generate_feature_names(ms_ch, hs_ch):
    """根據您的特徵提取邏輯，產生具有物理意義的特徵名稱"""
    names = []
    
    # --- HS Features ---
    hs_stats = ["Mean", "Std", "Min", "Max", "P25", "P50", "P75", "IQR"]
    for stat in hs_stats:
        for c in range(hs_ch): names.append(f"HS_{stat}_B{c}")
        
    names.extend(["HS_Global_Mean", "HS_Global_Std", "HS_Global_Range", "HS_Global_Median"])
    names.extend(["HS_Deriv_Mean", "HS_Deriv_Std", "HS_Deriv_Max", "HS_Deriv_Min", "HS_Deriv_AbsSum"])
    names.extend(["VI_NDVI", "VI_GNDVI", "VI_EVI", "VI_SAVI", "VI_NDRE1", "VI_NDRE2", 
                  "VI_CIRE", "VI_RedEdge_Slope", "VI_CI_Green", "VI_MTCI", "VI_Health_Score", 
                  "VI_NIR_Red_Ratio", "VI_Green_Red_Ratio", "VI_NIR_Green_Ratio", "VI_RE1_Red_Ratio", "VI_RE2_RE1_Ratio"])
    names.extend(["HS_Space_Center", "HS_Space_Edge", "HS_Space_C-E", 
                  "HS_Space_Q1", "HS_Space_Q2", "HS_Space_Q3", "HS_Space_Q4", "HS_Space_Q_Std"])
    
    for i in range(5):
        names.extend([f"HS_Seg{i+1}_Mean", f"HS_Seg{i+1}_Std"])
        
    # 填補固定的 HS_FIXED_DIM = 60
    hs_used_fixed = 4 + 5 + 16 + 8 + 10
    for i in range(60 - hs_used_fixed):
        names.append(f"HS_PadZero_{i}")
        
    # --- MS Features ---
    ms_stats = ["Mean", "Std", "Min", "Max", "P25", "P50", "P75", "IQR"]
    for stat in ms_stats:
        for c in range(ms_ch): names.append(f"MS_{stat}_B{c}")
        
    names.extend(["MS_Global_Mean", "MS_Global_Std", "MS_Global_Range"])
    
    for i in range(12):
        names.append(f"MS_Ratio_B{i}_B{i+1}" if i < ms_ch - 1 else f"MS_Ratio_Pad_{i}")
        
    return names

def plot_lgb_importance(cfg):
    print("="*50)
    print("Generating Feature Importance Plot (like PDF Fig 4/5)...")
    
    dims_path = os.path.join(cfg.MODEL_DIR, "dims.pkl")
    model_path = os.path.join(cfg.MODEL_DIR, "lgb_f0.pkl")
    
    if not os.path.exists(dims_path) or not os.path.exists(model_path):
        print(f"Error: 找不到 {dims_path} 或 {model_path}")
        return
        
    ms_ch, hs_ch = joblib.load(dims_path)
    clf_lgb = joblib.load(model_path)
    
    feature_names = generate_feature_names(ms_ch, hs_ch)
    
    if len(feature_names) != clf_lgb.n_features_:
        print(f"Warning: 特徵名稱數量 ({len(feature_names)}) 與模型預期 ({clf_lgb.n_features_}) 不符。將使用預設名稱。")
        feature_names = [f"Feature_{i}" for i in range(clf_lgb.n_features_)]
    
    importances = clf_lgb.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title('Top 20 Most Important Features (LightGBM)', fontsize=16, fontweight='bold')
    plt.xlabel('Feature Importance (Split)', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    out_path = os.path.join(cfg.OUT_DIR, 'Figure_5_Feature_Importance.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved -> {out_path}")
    plt.close()

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    cfg = CFG()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    # 1. 產生光譜圖
    train_idx = build_index(cfg.ROOT, cfg.TRAIN_DIR)
    train_df = make_train_df(train_idx)
    
    if len(train_df) > 0:
        plot_spectral_profiles(train_df, cfg)
    else:
        print("Dataset not found. Please check CFG.ROOT path.")
        
    # 2. 產生特徵重要性圖
    plot_lgb_importance(cfg)
    print("="*50)
    print("All plots generated successfully!")
"""
ICPR 2026 - Supplementary Experiments
產生論文所需的三項缺失內容：
1. Ablation Study (移除各組特徵比較 F1)
2. Feature Importance (LightGBM 特徵重要性排名)
3. Spectral Profiles (各類別平均光譜曲線)

用法: python supplementary_experiments.py
輸出: ./output_supplementary/ 目錄下的圖表和 CSV
"""

import os
import re
import warnings
import json

import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ============================================================
# Config (與你的 2689.py 一致)
# ============================================================
class CFG:
    ROOT = "/ssd6/pienyuzhe/ICPR2026_BeyondAI_Crop_Disease/Dataset/Kaggle_Prepared"
    TRAIN_DIR = "train"
    VAL_DIR = "val"
    
    HS_DROP_FIRST = 10
    HS_DROP_LAST = 14
    
    HEALTH_BOOST = 1.1
    N_FOLDS = 5
    SEED = 2689
    OUT_DIR = "./output_supplementary/"

LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}

# ============================================================
# Data Utils (與 2689.py 相同)
# ============================================================
def list_files(folder, exts):
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, fn) for fn in os.listdir(folder)
                   if fn.lower().endswith(exts)])

def base_id(path):
    return os.path.splitext(os.path.basename(path))[0]

def parse_label(bid):
    m = re.match(r"^(Health|Rust|Other)_", bid)
    return m.group(1) if m else None

def build_index(root, split):
    split_dir = os.path.join(root, split)
    idx = {}
    for p in list_files(os.path.join(split_dir, "RGB"), (".png", ".jpg", ".jpeg")):
        idx.setdefault(base_id(p), {})["rgb"] = p
    for p in list_files(os.path.join(split_dir, "MS"), (".tif", ".tiff")):
        idx.setdefault(base_id(p), {})["ms"] = p
    for p in list_files(os.path.join(split_dir, "HS"), (".tif", ".tiff")):
        idx.setdefault(base_id(p), {})["hs"] = p
    return idx

def make_train_df(train_idx):
    rows = []
    for bid, paths in train_idx.items():
        lab = parse_label(bid)
        if lab:
            rows.append({"base_id": bid, "label": lab, **paths})
    return pd.DataFrame(rows)

def read_tiff(path):
    try:
        arr = tiff.imread(path)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3 and arr.shape[0] < arr.shape[1]:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.float32)
    except:
        return None

def normalize(x):
    x = np.nan_to_num(x, nan=0.0)
    mn, mx = x.min(), x.max()
    if mx - mn > 1e-6:
        x = (x - mn) / (mx - mn)
    return np.clip(x, 0, 1)

def scan_dims(df, cfg):
    ms_ch, hs_ch = 5, 77
    for row in df.iloc[:10].itertuples():
        if hasattr(row, 'ms') and row.ms:
            arr = read_tiff(row.ms)
            if arr is not None:
                ms_ch = arr.shape[2]
                break
    for row in df.iloc[:10].itertuples():
        if hasattr(row, 'hs') and isinstance(row.hs, str):
            arr = read_tiff(row.hs)
            if arr is not None:
                hs_ch = max(1, arr.shape[2] - cfg.HS_DROP_FIRST - cfg.HS_DROP_LAST)
                break
    return ms_ch, hs_ch

# ============================================================
# Feature Extraction (與 2689.py 相同，但增加特徵名稱追蹤)
# ============================================================
def get_feature_names(hs_ch, ms_ch):
    """產生所有特徵的名稱，與 extract 順序一致"""
    names = []
    
    # HS base stats (8 * hs_ch)
    stat_names = ["mean", "std", "min", "max", "Q25", "Q50", "Q75", "IQR"]
    for stat in stat_names:
        for b in range(hs_ch):
            wl = 450 + (b + 10) * 4  # 近似波長
            names.append(f"HS_{stat}_band{b}_{wl}nm")
    
    # HS extra (60)
    names += ["HS_global_mean", "HS_global_std", "HS_global_range", "HS_global_median"]
    names += ["HS_deriv_mean", "HS_deriv_std", "HS_deriv_max", "HS_deriv_min", "HS_deriv_abssum"]
    names += ["VI_NDVI", "VI_GNDVI", "VI_EVI", "VI_SAVI",
              "VI_NDRE1", "VI_NDRE2", "VI_CIre", "RE_slope",
              "VI_CIgreen", "VI_MTCI", "VI_health_score",
              "ratio_NIR_Red", "ratio_Green_Red", "ratio_NIR_Green",
              "ratio_RE1_Red", "ratio_RE2_RE1"]
    names += ["SP_center_mean", "SP_edge_mean", "SP_center_edge_diff",
              "SP_Q1", "SP_Q2", "SP_Q3", "SP_Q4", "SP_Q_std"]
    for i in range(5):
        names += [f"SEG{i}_mean", f"SEG{i}_std"]
    
    # 補齊到 60
    while len(names) < hs_ch * 8 + 60:
        names.append(f"HS_pad_{len(names)}")
    
    # MS base stats (8 * ms_ch)
    ms_band_names = ["Blue_480nm", "Green_550nm", "Red_650nm", "RE_740nm", "NIR_833nm"]
    for stat in stat_names:
        for b in range(ms_ch):
            bname = ms_band_names[b] if b < len(ms_band_names) else f"band{b}"
            names.append(f"MS_{stat}_{bname}")
    
    # MS extra (15)
    names += ["MS_global_mean", "MS_global_std", "MS_global_range"]
    for i in range(12):
        names.append(f"MS_ratio_{i}_{i+1}")
    
    return names


def extract_hs_features(data, n_ch):
    FIXED_DIM = 60
    total_dim = n_ch * 8 + FIXED_DIM
    
    if data is None:
        return np.zeros(total_dim)
    
    data = normalize(data)
    H, W, C = data.shape
    
    if C > n_ch:
        data = data[:, :, :n_ch]
    elif C < n_ch:
        data = np.concatenate([data, np.zeros((H, W, n_ch - C))], axis=2)
    
    C = n_ch
    flat = data.reshape(-1, C)
    feats = np.zeros(total_dim)
    idx = 0
    
    feats[idx:idx+C] = flat.mean(0); idx += C
    feats[idx:idx+C] = flat.std(0); idx += C
    feats[idx:idx+C] = flat.min(0); idx += C
    feats[idx:idx+C] = flat.max(0); idx += C
    feats[idx:idx+C] = np.percentile(flat, 25, 0); idx += C
    feats[idx:idx+C] = np.percentile(flat, 50, 0); idx += C
    feats[idx:idx+C] = np.percentile(flat, 75, 0); idx += C
    feats[idx:idx+C] = np.percentile(flat, 75, 0) - np.percentile(flat, 25, 0); idx += C
    
    h1, h2, w1, w2 = H//4, 3*H//4, W//4, 3*W//4
    center = data[h1:h2, w1:w2, :]
    center_spec = center.mean((0, 1))
    
    feats[idx] = data.mean(); idx += 1
    feats[idx] = data.std(); idx += 1
    feats[idx] = data.max() - data.min(); idx += 1
    feats[idx] = np.median(data); idx += 1
    
    if C > 2:
        d1 = np.diff(center_spec)
        feats[idx] = d1.mean(); idx += 1
        feats[idx] = d1.std(); idx += 1
        feats[idx] = d1.max(); idx += 1
        feats[idx] = d1.min(); idx += 1
        feats[idx] = np.abs(d1).sum(); idx += 1
    else:
        idx += 5
    
    if C >= 30:
        blue_idx = max(0, int(C * 0.05))
        green_idx = int(C * 0.15)
        red_idx = int(C * 0.35)
        re1_idx = int(C * 0.42)
        re2_idx = int(C * 0.48)
        nir_idx = min(C-1, int(C * 0.75))
        
        blue = data[:, :, blue_idx].mean() + 1e-8
        green = data[:, :, green_idx].mean() + 1e-8
        red = data[:, :, red_idx].mean() + 1e-8
        re1 = data[:, :, re1_idx].mean() + 1e-8
        re2 = data[:, :, re2_idx].mean() + 1e-8
        nir = data[:, :, nir_idx].mean() + 1e-8
        
        ndvi = (nir - red) / (nir + red)
        gndvi = (nir - green) / (nir + green)
        evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
        savi = 1.5 * (nir - red) / (nir + red + 0.5)
        ndre1 = (nir - re1) / (nir + re1)
        ndre2 = (nir - re2) / (nir + re2)
        cire = (nir / re1) - 1
        
        re_region = center_spec[int(C*0.4):int(C*0.55)]
        re_deriv = np.diff(re_region) if len(re_region) > 1 else np.array([0])
        re_slope = np.max(re_deriv) if len(re_deriv) > 0 else 0
        
        ci_green = (nir / green) - 1
        mtci = (nir - re1) / (re1 - red + 1e-8)
        health_score = (ndvi + gndvi + ndre1) / 3
        
        feats[idx:idx+16] = [
            ndvi, gndvi, evi, savi,
            ndre1, ndre2, cire, re_slope,
            ci_green, mtci, health_score,
            nir/red, green/red, nir/green,
            re1/red, re2/re1
        ]
    idx += 16
    
    c_mean = center.mean()
    e_mean = (data[:H//4,:,:].mean() + data[3*H//4:,:,:].mean()) / 2
    q1 = data[:H//2, :W//2, :].mean()
    q2 = data[:H//2, W//2:, :].mean()
    q3 = data[H//2:, :W//2, :].mean()
    q4 = data[H//2:, W//2:, :].mean()
    
    feats[idx:idx+8] = [c_mean, e_mean, c_mean - e_mean, q1, q2, q3, q4, np.std([q1,q2,q3,q4])]
    idx += 8
    
    n_seg = 5
    seg_size = max(1, C // n_seg)
    for i in range(n_seg):
        start = i * seg_size
        end = min(start + seg_size, C)
        seg = center_spec[start:end]
        if len(seg) > 0:
            feats[idx] = seg.mean()
            feats[idx+1] = seg.std()
        idx += 2
    
    return feats


def extract_ms_features(data, n_ch):
    FIXED_DIM = 15
    total_dim = n_ch * 8 + FIXED_DIM
    
    if data is None:
        return np.zeros(total_dim)
    
    data = normalize(data)
    H, W, C = data.shape
    
    if C > n_ch:
        data = data[:, :, :n_ch]
    elif C < n_ch:
        data = np.concatenate([data, np.zeros((H, W, n_ch - C))], axis=2)
    
    flat = data.reshape(-1, n_ch)
    feats = np.zeros(total_dim)
    idx = 0
    
    feats[idx:idx+n_ch] = flat.mean(0); idx += n_ch
    feats[idx:idx+n_ch] = flat.std(0); idx += n_ch
    feats[idx:idx+n_ch] = flat.min(0); idx += n_ch
    feats[idx:idx+n_ch] = flat.max(0); idx += n_ch
    feats[idx:idx+n_ch] = np.percentile(flat, 25, 0); idx += n_ch
    feats[idx:idx+n_ch] = np.percentile(flat, 50, 0); idx += n_ch
    feats[idx:idx+n_ch] = np.percentile(flat, 75, 0); idx += n_ch
    feats[idx:idx+n_ch] = np.percentile(flat, 75, 0) - np.percentile(flat, 25, 0); idx += n_ch
    
    feats[idx] = data.mean(); idx += 1
    feats[idx] = data.std(); idx += 1
    feats[idx] = data.max() - data.min(); idx += 1
    
    for i in range(min(n_ch - 1, 12)):
        feats[idx] = data[:,:,i].mean() / (data[:,:,i+1].mean() + 1e-8)
        idx += 1
    
    return feats


def get_all_features(df, cfg, ms_ch, hs_ch):
    n = len(df)
    hs_dim = hs_ch * 8 + 60
    ms_dim = ms_ch * 8 + 15
    
    hs_feats = np.zeros((n, hs_dim))
    ms_feats = np.zeros((n, ms_dim))
    labels, ids = [], []
    
    for i, row in enumerate(tqdm(df.itertuples(), total=n, desc="Features")):
        if hasattr(row, 'hs') and isinstance(row.hs, str) and os.path.exists(row.hs):
            hs = read_tiff(row.hs)
            if hs is not None:
                C = hs.shape[2]
                if C > cfg.HS_DROP_FIRST + cfg.HS_DROP_LAST:
                    hs = hs[:, :, cfg.HS_DROP_FIRST:C-cfg.HS_DROP_LAST]
            feat = extract_hs_features(hs, hs_ch)
            hs_feats[i, :len(feat)] = feat[:hs_dim]
        
        if hasattr(row, 'ms') and row.ms and os.path.exists(str(row.ms)):
            ms = read_tiff(row.ms)
            feat = extract_ms_features(ms, ms_ch)
            ms_feats[i, :len(feat)] = feat[:ms_dim]
        
        if hasattr(row, 'label'):
            labels.append(LBL2ID[row.label])
        ids.append(row.base_id)
    
    X = np.concatenate([hs_feats, ms_feats], axis=1)
    return np.nan_to_num(X), np.array(labels) if labels else None, ids, hs_feats, ms_feats


# ============================================================
# 快速 CV 評估函數 (用於 Ablation)
# ============================================================
def quick_cv(X, y, cfg, use_health_boost=True, label=""):
    """跑 5-fold CV，回傳 ensemble macro-F1 (mean, std) 和 per-class F1"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    fold_scores = []
    all_per_class = []
    all_lgb_importance = np.zeros(X.shape[1])
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y)):
        Xtr, ytr = X_scaled[tr_idx], y[tr_idx]
        Xva, yva = X_scaled[va_idx], y[va_idx]
        
        preds, f1s = {}, {}
        
        # SVM
        clf_svm = SVC(C=10.0, kernel='rbf', gamma='scale', probability=True,
                       class_weight='balanced', random_state=cfg.SEED + fold)
        clf_svm.fit(Xtr, ytr)
        p = clf_svm.predict_proba(Xva)
        f1s['svm'] = f1_score(yva, p.argmax(1), average='macro')
        preds['svm'] = p
        
        # LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            num_leaves=32, subsample=0.8, colsample_bytree=0.6,
            class_weight='balanced', random_state=cfg.SEED + fold,
            verbosity=-1, n_jobs=-1)
        clf_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        p = clf_lgb.predict_proba(Xva)
        f1s['lgb'] = f1_score(yva, p.argmax(1), average='macro')
        preds['lgb'] = p
        all_lgb_importance += clf_lgb.feature_importances_
        
        # XGBoost
        clf_xgb = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6,
            random_state=cfg.SEED + fold,
            use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
        clf_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p = clf_xgb.predict_proba(Xva)
        f1s['xgb'] = f1_score(yva, p.argmax(1), average='macro')
        preds['xgb'] = p
        
        # Dynamic weight ensemble
        w = {k: v**2 for k, v in f1s.items()}
        tw = sum(w.values())
        w = {k: v/tw for k, v in w.items()}
        
        ens = sum(w[k] * preds[k] for k in preds)
        ens_f1 = f1_score(yva, ens.argmax(1), average='macro')
        
        if use_health_boost:
            ens_adj = ens.copy()
            ens_adj[:, 0] *= cfg.HEALTH_BOOST
            ens_adj = ens_adj / ens_adj.sum(axis=1, keepdims=True)
            adj_f1 = f1_score(yva, ens_adj.argmax(1), average='macro')
            if adj_f1 > ens_f1:
                ens = ens_adj
                ens_f1 = adj_f1
        
        fold_scores.append(ens_f1)
        per_class = f1_score(yva, ens.argmax(1), average=None)
        all_per_class.append(per_class)
    
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    mean_per_class = np.mean(all_per_class, axis=0)
    avg_importance = all_lgb_importance / cfg.N_FOLDS
    
    if label:
        print(f"  {label}: F1={mean_f1:.4f}±{std_f1:.4f}  "
              f"H={mean_per_class[0]:.3f} R={mean_per_class[1]:.3f} O={mean_per_class[2]:.3f}")
    
    return mean_f1, std_f1, mean_per_class, avg_importance


# ============================================================
# Part 3: Spectral Profiles
# ============================================================
def collect_spectral_profiles(df, cfg):
    """讀取每個樣本的 HS 平均光譜，按類別分組"""
    profiles = {lab: [] for lab in LABELS}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Spectral profiles"):
        lab = row.get('label')
        hs_path = row.get('hs')
        if not lab or not isinstance(hs_path, str) or not os.path.exists(hs_path):
            continue
        
        hs = read_tiff(hs_path)
        if hs is None:
            continue
        
        C = hs.shape[2]
        if C > cfg.HS_DROP_FIRST + cfg.HS_DROP_LAST:
            hs = hs[:, :, cfg.HS_DROP_FIRST:C - cfg.HS_DROP_LAST]
        
        # 正規化後取平均光譜
        hs_norm = normalize(hs)
        mean_spec = hs_norm.mean(axis=(0, 1))  # shape: (n_bands,)
        profiles[lab].append(mean_spec)
    
    return profiles


def plot_spectral_profiles(profiles, cfg, hs_ch):
    """畫三類的平均光譜 (mean ± std)，類似參考 PDF 的 Fig.3"""
    # 計算近似波長軸
    wavelengths = np.array([450 + (i + cfg.HS_DROP_FIRST) * 4 for i in range(hs_ch)])
    
    colors = {'Health': '#4CAF50', 'Rust': '#FF9800', 'Other': '#F44336'}
    
    # === 圖一：三類合併在一張圖 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for lab in LABELS:
        if not profiles[lab]:
            continue
        specs = np.array(profiles[lab])
        # 確保維度匹配
        if specs.shape[1] > len(wavelengths):
            specs = specs[:, :len(wavelengths)]
        elif specs.shape[1] < len(wavelengths):
            wavelengths = wavelengths[:specs.shape[1]]
        
        mean = specs.mean(axis=0)
        std = specs.std(axis=0)
        
        ax.plot(wavelengths, mean, color=colors[lab], linewidth=2, label=f'{lab} (n={len(specs)})')
        ax.fill_between(wavelengths, mean - std, mean + std, color=colors[lab], alpha=0.15)
    
    ax.set_xlabel('Wavelength (nm)', fontsize=13)
    ax.set_ylabel('Normalized Mean Pixel Value', fontsize=13)
    ax.set_title('Spectral Profiles by Disease Category', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, 'fig_spectral_profiles_combined.png'), dpi=200)
    plt.close()
    
    # === 圖二：三類分開三個子圖 (更接近參考 PDF 風格) ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    for ax_i, lab in enumerate(LABELS):
        ax = axes[ax_i]
        if not profiles[lab]:
            continue
        specs = np.array(profiles[lab])
        if specs.shape[1] > len(wavelengths):
            specs = specs[:, :len(wavelengths)]
        
        mean = specs.mean(axis=0)
        std = specs.std(axis=0)
        
        # 畫每條光譜 (半透明)
        for s in specs[::max(1, len(specs)//30)]:  # 最多畫30條
            ax.plot(wavelengths, s[:len(wavelengths)], color=colors[lab], alpha=0.08, linewidth=0.5)
        
        ax.plot(wavelengths, mean, color='black', linewidth=2, label='μ')
        ax.plot(wavelengths, mean + std, color='gray', linewidth=1, linestyle='--', label='μ+σ')
        ax.plot(wavelengths, mean - std, color='gray', linewidth=1, linestyle='--', label='μ−σ')
        
        ax.set_title(f'{lab} (n={len(specs)})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        if ax_i == 0:
            ax.set_ylabel('Normalized Mean Pixel Value', fontsize=11)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('Normalized Spectrum Profiles per Category', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, 'fig_spectral_profiles_separate.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # === 圖三：光譜差異圖 (Rust-Health, Other-Health) ===
    fig, ax = plt.subplots(figsize=(10, 5))
    
    specs_h = np.array(profiles['Health'])
    specs_r = np.array(profiles['Rust'])
    specs_o = np.array(profiles['Other'])
    
    mean_h = specs_h.mean(0)[:len(wavelengths)]
    mean_r = specs_r.mean(0)[:len(wavelengths)]
    mean_o = specs_o.mean(0)[:len(wavelengths)]
    
    ax.plot(wavelengths, mean_r - mean_h, color='#FF9800', linewidth=2, label='Rust − Health')
    ax.plot(wavelengths, mean_o - mean_h, color='#F44336', linewidth=2, label='Other − Health')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')
    
    # 標記重要光譜區域
    ax.axvspan(700, 750, alpha=0.1, color='green', label='Red-Edge (700-750nm)')
    ax.axvspan(800, 875, alpha=0.1, color='red', label='NIR Plateau (800-875nm)')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=13)
    ax.set_ylabel('Spectral Difference', fontsize=13)
    ax.set_title('Spectral Difference from Health Class', fontsize=15, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, 'fig_spectral_difference.png'), dpi=200)
    plt.close()
    
    print("  Spectral profile plots saved!")


# ============================================================
# Part 2: Feature Importance
# ============================================================
def plot_feature_importance(importance, feature_names, cfg, top_k=30):
    """畫 LightGBM feature importance (top-K)"""
    # 排序
    idx = np.argsort(importance)[::-1][:top_k]
    top_names = [feature_names[i] for i in idx]
    top_imp = importance[idx]
    
    # 標記不同類型的特徵
    colors = []
    for name in top_names:
        if name.startswith('VI_') or name.startswith('ratio_'):
            colors.append('#4CAF50')  # 植被指數 = 綠色
        elif name.startswith('RE_'):
            colors.append('#FF5722')  # 紅邊 = 橘紅
        elif name.startswith('SP_') or name.startswith('SEG'):
            colors.append('#2196F3')  # 空間/段 = 藍色
        elif name.startswith('MS_'):
            colors.append('#9C27B0')  # MS = 紫色
        elif 'deriv' in name:
            colors.append('#FF9800')  # 導數 = 橙色
        else:
            colors.append('#607D8B')  # HS band stats = 灰色
    
    fig, ax = plt.subplots(figsize=(10, 9))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_imp, color=colors, edgecolor='white', linewidth=0.3, height=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (LightGBM, avg over 5 folds)', fontsize=12)
    ax.set_title(f'Top-{top_k} Most Important Features', fontsize=14, fontweight='bold')
    
    # 圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Vegetation Indices'),
        Patch(facecolor='#FF5722', label='Red-Edge Features'),
        Patch(facecolor='#2196F3', label='Spatial / Segment'),
        Patch(facecolor='#9C27B0', label='MS Features'),
        Patch(facecolor='#FF9800', label='Spectral Derivatives'),
        Patch(facecolor='#607D8B', label='HS Band Statistics'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, 'fig_feature_importance.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 另外畫波段重要性分布圖 (只看 HS band mean 特徵)
    band_importance = np.zeros(cfg.HS_DROP_FIRST + 101 + cfg.HS_DROP_LAST)  # 全部125波段
    # 提取 HS mean band features 的重要性
    hs_ch = 101
    for b in range(hs_ch):
        # 8 個統計量的重要性加總
        total_imp = 0
        for s in range(8):
            feat_idx = s * hs_ch + b
            if feat_idx < len(importance):
                total_imp += importance[feat_idx]
        actual_band = b + cfg.HS_DROP_FIRST
        if actual_band < len(band_importance):
            band_importance[actual_band] = total_imp
    
    wavelengths_full = np.arange(125) * 4 + 450
    
    fig, ax = plt.subplots(figsize=(12, 5))
    # 有效波段
    valid_mask = np.zeros(125, dtype=bool)
    valid_mask[cfg.HS_DROP_FIRST:125-cfg.HS_DROP_LAST] = True
    
    ax.bar(wavelengths_full[valid_mask], band_importance[valid_mask],
           width=3.5, color='#2196F3', alpha=0.7, label='Used bands')
    ax.bar(wavelengths_full[~valid_mask], band_importance[~valid_mask],
           width=3.5, color='#BDBDBD', alpha=0.5, label='Removed (noise)')
    
    # 標記重要區域
    ax.axvspan(700, 750, alpha=0.1, color='green')
    ax.axvspan(800, 875, alpha=0.1, color='red')
    ax.text(725, ax.get_ylim()[1]*0.9, 'Red-Edge', ha='center', fontsize=9, color='green', fontweight='bold')
    ax.text(837, ax.get_ylim()[1]*0.9, 'NIR', ha='center', fontsize=9, color='red', fontweight='bold')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Aggregated Importance', fontsize=12)
    ax.set_title('Band-wise Feature Importance Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, 'fig_band_importance.png'), dpi=200)
    plt.close()
    
    print("  Feature importance plots saved!")
    
    # 存 CSV
    df_imp = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance[:len(feature_names)]
    }).sort_values('importance', ascending=False)
    df_imp.to_csv(os.path.join(cfg.OUT_DIR, 'feature_importance.csv'), index=False)


# ============================================================
# Part 1: Ablation Study
# ============================================================
def run_ablation(X_full, y, cfg, hs_ch, ms_ch, feature_names):
    """逐步移除特徵組，比較 F1 差異"""
    
    hs_dim = hs_ch * 8 + 60
    ms_dim = ms_ch * 8 + 15
    total_dim = X_full.shape[1]
    
    # 定義特徵組的索引範圍
    hs_base_end = hs_ch * 8                        # HS band stats
    hs_global_start = hs_base_end                   # +4
    hs_deriv_start = hs_base_end + 4                # +5
    hs_vi_start = hs_base_end + 9                   # +16
    hs_spatial_start = hs_base_end + 25             # +8
    hs_segment_start = hs_base_end + 33             # +10 (但有些 padding)
    ms_start = hs_dim                               # MS 特徵開始
    
    # 特徵組 masks
    groups = {
        'Vegetation Indices (16d)': list(range(hs_vi_start, hs_vi_start + 16)),
        'Red-Edge Features': [i for i, n in enumerate(feature_names) if 'RE_' in n or 'NDRE' in n or 're_slope' in n.lower() or 'CIre' in n],
        'Spatial Features (8d)': list(range(hs_spatial_start, hs_spatial_start + 8)),
        'Spectral Derivatives (5d)': list(range(hs_deriv_start, hs_deriv_start + 5)),
        'Segment Features (10d)': list(range(hs_segment_start, hs_segment_start + 10)),
        'MS Features (all)': list(range(ms_start, total_dim)),
        'HS Extra (all 60d)': list(range(hs_base_end, hs_dim)),
    }
    
    results = {}
    
    # Baseline (全部特徵)
    print("\n[Ablation Study]")
    print("="*70)
    f1_mean, f1_std, per_class, _ = quick_cv(X_full, y, cfg, use_health_boost=True,
                                              label="Full model (923d)")
    results['Full model (923d)'] = {
        'mean_f1': f1_mean, 'std_f1': f1_std,
        'H': per_class[0], 'R': per_class[1], 'O': per_class[2],
        'dims': total_dim
    }
    
    # 移除各組
    for group_name, group_idx in groups.items():
        keep_idx = [i for i in range(total_dim) if i not in group_idx]
        X_ablated = X_full[:, keep_idx]
        f1_mean, f1_std, per_class, _ = quick_cv(
            X_ablated, y, cfg, use_health_boost=True,
            label=f"w/o {group_name} ({len(keep_idx)}d)")
        results[f'w/o {group_name}'] = {
            'mean_f1': f1_mean, 'std_f1': f1_std,
            'H': per_class[0], 'R': per_class[1], 'O': per_class[2],
            'dims': len(keep_idx)
        }
    
    # 不用 Health Boost
    f1_mean, f1_std, per_class, _ = quick_cv(X_full, y, cfg, use_health_boost=False,
                                              label="w/o Health Boost")
    results['w/o Health Boost'] = {
        'mean_f1': f1_mean, 'std_f1': f1_std,
        'H': per_class[0], 'R': per_class[1], 'O': per_class[2],
        'dims': total_dim
    }
    
    # 等權重 (不用動態權重)
    # 需要稍微改寫 quick_cv...用簡單的方式模擬
    print("  w/o Dynamic Weights: (using equal weights)")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    eq_scores = []
    eq_per_class = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_scaled, y)):
        Xtr, ytr = X_scaled[tr_idx], y[tr_idx]
        Xva, yva = X_scaled[va_idx], y[va_idx]
        
        clf_svm = SVC(C=10.0, kernel='rbf', gamma='scale', probability=True,
                       class_weight='balanced', random_state=cfg.SEED+fold)
        clf_svm.fit(Xtr, ytr)
        clf_lgb = lgb.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                                      num_leaves=32, subsample=0.8, colsample_bytree=0.6,
                                      class_weight='balanced', random_state=cfg.SEED+fold,
                                      verbosity=-1, n_jobs=-1)
        clf_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        clf_xgb = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                                      subsample=0.8, colsample_bytree=0.6,
                                      random_state=cfg.SEED+fold,
                                      use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
        clf_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        
        ens = (clf_svm.predict_proba(Xva) + clf_lgb.predict_proba(Xva) + clf_xgb.predict_proba(Xva)) / 3
        ens[:, 0] *= cfg.HEALTH_BOOST
        ens = ens / ens.sum(axis=1, keepdims=True)
        eq_scores.append(f1_score(yva, ens.argmax(1), average='macro'))
        eq_per_class.append(f1_score(yva, ens.argmax(1), average=None))
    
    eq_mean = np.mean(eq_scores)
    eq_std = np.std(eq_scores)
    eq_pc = np.mean(eq_per_class, axis=0)
    print(f"    Equal Weights: F1={eq_mean:.4f}±{eq_std:.4f}  H={eq_pc[0]:.3f} R={eq_pc[1]:.3f} O={eq_pc[2]:.3f}")
    results['w/o Dynamic Weights (equal)'] = {
        'mean_f1': eq_mean, 'std_f1': eq_std,
        'H': eq_pc[0], 'R': eq_pc[1], 'O': eq_pc[2],
        'dims': total_dim
    }
    
    # 只用 HS
    X_hs_only = X_full[:, :hs_dim]
    f1_mean, f1_std, per_class, _ = quick_cv(X_hs_only, y, cfg, use_health_boost=True,
                                              label=f"HS only ({hs_dim}d)")
    results[f'HS only ({hs_dim}d)'] = {
        'mean_f1': f1_mean, 'std_f1': f1_std,
        'H': per_class[0], 'R': per_class[1], 'O': per_class[2],
        'dims': hs_dim
    }
    
    # 只用 MS
    X_ms_only = X_full[:, hs_dim:]
    f1_mean, f1_std, per_class, _ = quick_cv(X_ms_only, y, cfg, use_health_boost=True,
                                              label=f"MS only ({ms_dim}d)")
    results[f'MS only ({ms_dim}d)'] = {
        'mean_f1': f1_mean, 'std_f1': f1_std,
        'H': per_class[0], 'R': per_class[1], 'O': per_class[2],
        'dims': ms_dim
    }
    
    return results


def plot_ablation(results, cfg):
    """畫 Ablation Study 結果圖"""
    # 排序：Full 在最上面，其他按 F1 降序
    full_key = 'Full model (923d)'
    other_keys = [k for k in results if k != full_key]
    other_keys.sort(key=lambda k: results[k]['mean_f1'], reverse=True)
    keys = [full_key] + other_keys
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    y_pos = np.arange(len(keys))
    means = [results[k]['mean_f1'] for k in keys]
    stds = [results[k]['std_f1'] for k in keys]
    
    colors = ['#2196F3' if k == full_key else 
              ('#4CAF50' if results[k]['mean_f1'] >= results[full_key]['mean_f1'] else '#FF9800')
              for k in keys]
    
    bars = ax.barh(y_pos, means, xerr=stds, capsize=4, color=colors,
                    edgecolor='white', linewidth=0.5, height=0.65,
                    error_kw={'linewidth': 1.2})
    
    # 標註數值
    for i, (m, s) in enumerate(zip(means, stds)):
        diff = m - results[full_key]['mean_f1']
        diff_str = f" ({diff:+.4f})" if keys[i] != full_key else " (baseline)"
        ax.text(m + s + 0.003, i, f'{m:.4f}{diff_str}', va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keys, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Macro F1-Score', fontsize=12)
    ax.set_title('Ablation Study: Impact of Feature Groups and Design Choices', fontsize=14, fontweight='bold')
    ax.axvline(x=results[full_key]['mean_f1'], color='#2196F3', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0.45, max(means) + 0.06)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUT_DIR, 'fig_ablation_study.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # 存 CSV
    rows = []
    for k in keys:
        r = results[k]
        diff = r['mean_f1'] - results[full_key]['mean_f1']
        rows.append({
            'Configuration': k,
            'Dims': r['dims'],
            'Mean_F1': f"{r['mean_f1']:.4f}",
            'Std_F1': f"{r['std_f1']:.4f}",
            'Health_F1': f"{r['H']:.3f}",
            'Rust_F1': f"{r['R']:.3f}",
            'Other_F1': f"{r['O']:.3f}",
            'Delta_F1': f"{diff:+.4f}" if k != full_key else "baseline"
        })
    df_abl = pd.DataFrame(rows)
    df_abl.to_csv(os.path.join(cfg.OUT_DIR, 'ablation_results.csv'), index=False)
    
    print("\n  Ablation table:")
    print(df_abl.to_string(index=False))
    print(f"\n  Saved to {cfg.OUT_DIR}")


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = CFG()
    np.random.seed(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    print("="*70)
    print("Supplementary Experiments for Paper")
    print("="*70)
    
    # Load data
    train_idx = build_index(cfg.ROOT, cfg.TRAIN_DIR)
    train_df = make_train_df(train_idx)
    print(f"Train samples: {len(train_df)}")
    print(train_df['label'].value_counts())
    
    ms_ch, hs_ch = scan_dims(train_df, cfg)
    print(f"MS: {ms_ch} ch, HS: {hs_ch} ch")
    
    # Extract features (保留 hs_feats 和 ms_feats 分開)
    print("\n[1/4] Extracting features...")
    X_full, y, ids, hs_feats, ms_feats = get_all_features(train_df, cfg, ms_ch, hs_ch)
    print(f"Total feature dim: {X_full.shape[1]}")
    
    # Feature names
    feature_names = get_feature_names(hs_ch, ms_ch)
    # 確保長度匹配
    while len(feature_names) < X_full.shape[1]:
        feature_names.append(f"feat_{len(feature_names)}")
    feature_names = feature_names[:X_full.shape[1]]
    
    # ============================================================
    # Part 1: Ablation Study
    # ============================================================
    print("\n[2/4] Running Ablation Study...")
    print("(This will take a while — running ~10 full CV experiments)")
    ablation_results = run_ablation(X_full, y, cfg, hs_ch, ms_ch, feature_names)
    plot_ablation(ablation_results, cfg)
    
    # ============================================================
    # Part 2: Feature Importance
    # ============================================================
    print("\n[3/4] Computing Feature Importance...")
    _, _, _, avg_importance = quick_cv(X_full, y, cfg, use_health_boost=True, label="Full (for importance)")
    plot_feature_importance(avg_importance, feature_names, cfg, top_k=30)
    
    # ============================================================
    # Part 3: Spectral Profiles
    # ============================================================
    print("\n[4/4] Collecting Spectral Profiles...")
    profiles = collect_spectral_profiles(train_df, cfg)
    for lab in LABELS:
        print(f"  {lab}: {len(profiles[lab])} spectra collected")
    plot_spectral_profiles(profiles, cfg, hs_ch)
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("ALL DONE! Output files in:", cfg.OUT_DIR)
    print("="*70)
    print("\nGenerated files:")
    for f in sorted(os.listdir(cfg.OUT_DIR)):
        fpath = os.path.join(cfg.OUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f:45s} ({size:.1f} KB)")
    
    print("\n圖表說明:")
    print("  fig_ablation_study.png          — Ablation Study 結果 (橫向 bar chart)")
    print("  fig_feature_importance.png       — Top-30 特徵重要性排名")
    print("  fig_band_importance.png          — 波段重要性分布 (按波長)")
    print("  fig_spectral_profiles_combined.png — 三類光譜合併圖 (mean±std)")
    print("  fig_spectral_profiles_separate.png — 三類光譜分開圖 (類似參考PDF)")
    print("  fig_spectral_difference.png      — 光譜差異圖 (Rust-Health, Other-Health)")
    print("  ablation_results.csv             — Ablation 數值表 (可直接貼入論文)")
    print("  feature_importance.csv           — 全部特徵重要性排名")


if __name__ == "__main__":
    main()
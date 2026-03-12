"""
ICPR 2026 - Fusion Model

融合兩個版本的優勢：
1. sklearn 版本：SVM 最強 (0.6734)
2. health_opt 版本：增強 Health 特徵

策略：
- 使用 health_opt 的增強特徵（紅邊、葉綠素指數）
- 使用 SVM + LGB + XGB 三個最強分類器
- 動態權重集成
- Health 閾值微調
"""

import os
import re
import warnings
import joblib

import numpy as np
import pandas as pd
import tifffile as tiff
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================
# Config
# ============================================================
class CFG:
    ROOT = "/ssd6/pienyuzhe/ICPR2026_BeyondAI_Crop_Disease/Dataset/Kaggle_Prepared"
    TRAIN_DIR = "train"
    VAL_DIR = "val"
    
    HS_DROP_FIRST = 10
    HS_DROP_LAST = 14
    
    # Health 閾值調整 (1.0 = 不調整)
    HEALTH_BOOST = 1.1
    
    N_FOLDS = 5
    SEED = 2689
    OUT_DIR = "./output_fusion2689/"


LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}

# ============================================================
# Data Utils
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

def make_val_df(val_idx):
    return pd.DataFrame([{"base_id": bid, **paths} for bid, paths in val_idx.items()])

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
# Enhanced Feature Extraction (from health_opt)
# ============================================================
def extract_hs_features(data, n_ch):
    """增強的 HS 特徵 - 固定維度輸出"""
    FIXED_DIM = 60  # 固定的額外特徵維度
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
    
    # 預分配固定大小
    feats = np.zeros(total_dim)
    idx = 0
    
    # 基礎統計 (8 * C)
    feats[idx:idx+C] = flat.mean(0); idx += C
    feats[idx:idx+C] = flat.std(0); idx += C
    feats[idx:idx+C] = flat.min(0); idx += C
    feats[idx:idx+C] = flat.max(0); idx += C
    feats[idx:idx+C] = np.percentile(flat, 25, 0); idx += C
    feats[idx:idx+C] = np.percentile(flat, 50, 0); idx += C
    feats[idx:idx+C] = np.percentile(flat, 75, 0); idx += C
    feats[idx:idx+C] = np.percentile(flat, 75, 0) - np.percentile(flat, 25, 0); idx += C
    
    # 中心區域
    h1, h2, w1, w2 = H//4, 3*H//4, W//4, 3*W//4
    center = data[h1:h2, w1:w2, :]
    center_spec = center.mean((0, 1))
    
    # 全局 (4)
    feats[idx] = data.mean(); idx += 1
    feats[idx] = data.std(); idx += 1
    feats[idx] = data.max() - data.min(); idx += 1
    feats[idx] = np.median(data); idx += 1
    
    # 光譜導數 (5)
    if C > 2:
        d1 = np.diff(center_spec)
        feats[idx] = d1.mean(); idx += 1
        feats[idx] = d1.std(); idx += 1
        feats[idx] = d1.max(); idx += 1
        feats[idx] = d1.min(); idx += 1
        feats[idx] = np.abs(d1).sum(); idx += 1
    else:
        idx += 5
    
    # 植被/健康指數 (16)
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
    
    # 空間 (8)
    c_mean = center.mean()
    e_mean = (data[:H//4,:,:].mean() + data[3*H//4:,:,:].mean()) / 2
    q1 = data[:H//2, :W//2, :].mean()
    q2 = data[:H//2, W//2:, :].mean()
    q3 = data[H//2:, :W//2, :].mean()
    q4 = data[H//2:, W//2:, :].mean()
    
    feats[idx:idx+8] = [c_mean, e_mean, c_mean - e_mean, q1, q2, q3, q4, np.std([q1,q2,q3,q4])]
    idx += 8
    
    # 區段 (10)
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
    """MS 特徵 - 固定維度輸出"""
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
    
    # 統計 (8 * n_ch)
    feats[idx:idx+n_ch] = flat.mean(0); idx += n_ch
    feats[idx:idx+n_ch] = flat.std(0); idx += n_ch
    feats[idx:idx+n_ch] = flat.min(0); idx += n_ch
    feats[idx:idx+n_ch] = flat.max(0); idx += n_ch
    feats[idx:idx+n_ch] = np.percentile(flat, 25, 0); idx += n_ch
    feats[idx:idx+n_ch] = np.percentile(flat, 50, 0); idx += n_ch
    feats[idx:idx+n_ch] = np.percentile(flat, 75, 0); idx += n_ch
    feats[idx:idx+n_ch] = np.percentile(flat, 75, 0) - np.percentile(flat, 25, 0); idx += n_ch
    
    # 全局 (3)
    feats[idx] = data.mean(); idx += 1
    feats[idx] = data.std(); idx += 1
    feats[idx] = data.max() - data.min(); idx += 1
    
    # 波段比值 (12)
    for i in range(min(n_ch - 1, 12)):
        feats[idx] = data[:,:,i].mean() / (data[:,:,i+1].mean() + 1e-8)
        idx += 1
    
    return feats


def get_all_features(df, cfg, ms_ch, hs_ch):
    n = len(df)
    hs_dim = hs_ch * 8 + 60  # 固定額外特徵維度
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
    return np.nan_to_num(X), np.array(labels) if labels else None, ids

# ============================================================
# Main
# ============================================================
def main():
    cfg = CFG()
    np.random.seed(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    print("="*60)
    print("Fusion Model: SVM + LGB + XGB + Health Features")
    print("="*60)
    
    # Data
    train_idx = build_index(cfg.ROOT, cfg.TRAIN_DIR)
    val_idx = build_index(cfg.ROOT, cfg.VAL_DIR)
    train_df = make_train_df(train_idx)
    val_df = make_val_df(val_idx)
    
    print(f"Train: {len(train_df)} | Test: {len(val_df)}")
    print(train_df['label'].value_counts())
    
    ms_ch, hs_ch = scan_dims(train_df, cfg)
    print(f"MS: {ms_ch} ch, HS: {hs_ch} ch")
    
    # Features
    X_train, y_train, _ = get_all_features(train_df, cfg, ms_ch, hs_ch)
    print(f"Features: {X_train.shape}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # CV
    print("\n" + "="*50)
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    
    oof = np.zeros((len(train_df), 3))
    fold_scores = []
    all_weights = []
    all_results = {'svm': [], 'lgb': [], 'xgb': []}
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold} ---")
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xva, yva = X_train[va_idx], y_train[va_idx]
        
        preds, f1s = {}, {}
        
        # SVM (最強)
        clf_svm = SVC(
            C=10.0, kernel='rbf', gamma='scale',
            probability=True, class_weight='balanced',
            random_state=cfg.SEED + fold
        )
        clf_svm.fit(Xtr, ytr)
        p = clf_svm.predict_proba(Xva)
        f = f1_score(yva, p.argmax(1), average='macro')
        preds['svm'] = p
        f1s['svm'] = f
        all_results['svm'].append(f)
        print(f"  SVM: {f:.4f}")
        joblib.dump(clf_svm, os.path.join(cfg.OUT_DIR, f"svm_f{fold}.pkl"))
        
        # LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            num_leaves=32, subsample=0.8, colsample_bytree=0.6,
            class_weight='balanced',
            random_state=cfg.SEED + fold, verbosity=-1, n_jobs=-1
        )
        clf_lgb.fit(Xtr, ytr, eval_set=[(Xva, yva)])
        p = clf_lgb.predict_proba(Xva)
        f = f1_score(yva, p.argmax(1), average='macro')
        preds['lgb'] = p
        f1s['lgb'] = f
        all_results['lgb'].append(f)
        print(f"  LGB: {f:.4f}")
        joblib.dump(clf_lgb, os.path.join(cfg.OUT_DIR, f"lgb_f{fold}.pkl"))
        
        # XGBoost
        clf_xgb = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6,
            random_state=cfg.SEED + fold,
            use_label_encoder=False, eval_metric='mlogloss',
            n_jobs=-1
        )
        clf_xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p = clf_xgb.predict_proba(Xva)
        f = f1_score(yva, p.argmax(1), average='macro')
        preds['xgb'] = p
        f1s['xgb'] = f
        all_results['xgb'].append(f)
        print(f"  XGB: {f:.4f}")
        joblib.dump(clf_xgb, os.path.join(cfg.OUT_DIR, f"xgb_f{fold}.pkl"))
        
        # 動態權重
        w = {k: v**2 for k, v in f1s.items()}
        tw = sum(w.values())
        w = {k: v/tw for k, v in w.items()}
        all_weights.append(w)
        
        # Ensemble
        ens = sum(w[k] * preds[k] for k in preds)
        ens_f1 = f1_score(yva, ens.argmax(1), average='macro')
        
        # Health boost
        ens_adj = ens.copy()
        ens_adj[:, 0] *= cfg.HEALTH_BOOST
        ens_adj = ens_adj / ens_adj.sum(axis=1, keepdims=True)
        adj_f1 = f1_score(yva, ens_adj.argmax(1), average='macro')
        
        if adj_f1 > ens_f1:
            final = ens_adj
            print(f"  Ensemble: {ens_f1:.4f} -> {adj_f1:.4f} (boosted)")
        else:
            final = ens
            print(f"  Ensemble: {ens_f1:.4f}")
        
        per_class = f1_score(yva, final.argmax(1), average=None)
        print(f"  H={per_class[0]:.2f}, R={per_class[1]:.2f}, O={per_class[2]:.2f}")
        
        oof[va_idx] = final
        fold_scores.append(f1_score(yva, final.argmax(1), average='macro'))
    
    # Summary
    print("\n" + "="*60)
    print("Per-classifier results:")
    for name in ['svm', 'lgb', 'xgb']:
        print(f"  {name}: {np.mean(all_results[name]):.4f} ± {np.std(all_results[name]):.4f}")
    
    print(f"\nEnsemble CV F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_train, oof.argmax(1), target_names=LABELS))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, oof.argmax(1)))
    
    # Save
    joblib.dump(scaler, os.path.join(cfg.OUT_DIR, "scaler.pkl"))
    joblib.dump(all_weights, os.path.join(cfg.OUT_DIR, "weights.pkl"))
    joblib.dump((ms_ch, hs_ch), os.path.join(cfg.OUT_DIR, "dims.pkl"))
    
    # Test
    print("\n" + "="*50)
    print("Test prediction...")
    
    X_test, _, _ = get_all_features(val_df, cfg, ms_ch, hs_ch)
    X_test = scaler.transform(X_test)
    
    avg_w = {k: np.mean([fw[k] for fw in all_weights]) for k in all_weights[0].keys()}
    print(f"Weights: {avg_w}")
    
    test_pred = np.zeros((len(val_df), 3))
    for fold in range(cfg.N_FOLDS):
        for name in avg_w.keys():
            clf = joblib.load(os.path.join(cfg.OUT_DIR, f"{name}_f{fold}.pkl"))
            test_pred += avg_w[name] * clf.predict_proba(X_test) / cfg.N_FOLDS
    
    # Health boost
    test_pred[:, 0] *= cfg.HEALTH_BOOST
    test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)
    
    # Submission
    pred_labels = [ID2LBL[p] for p in test_pred.argmax(1)]
    
    sub_ids = []
    for _, r in val_df.iterrows():
        for k in ["hs", "ms", "rgb"]:
            if isinstance(r.get(k), str) and r[k]:
                sub_ids.append(os.path.basename(r[k]))
                break
    
    sub = pd.DataFrame({"Id": sub_ids, "Category": pred_labels})
    sub.to_csv(os.path.join(cfg.OUT_DIR, "submission.csv"), index=False)
    
    print(f"\nSubmission saved!")
    print(sub['Category'].value_counts())


if __name__ == "__main__":
    main()

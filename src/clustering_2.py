# cluster_hyperparam_search_v2.py
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src import config  # NEW: central config for paths

# ============================================================
# CONFIG
# ============================================================

FEATURE_SETS = {
    "A": ["mean_intensity", "std_intensity", "max_intensity", "ts_pca1"],
    "B": ["mean_intensity", "std_intensity", "max_intensity", "skew_intensity", "ts_pca1"],
}

K_VALUES = [3]
SPATIAL_WEIGHTS = [0.0, 0.2, 0.4, 0.6]
SMOOTH_ITERATIONS = [0, 1, 2]
ALGORITHMS = ["kmeans", "gmm_full"]
RANDOM_STATE = 42
PRIMARY = "spatial"

# --- Path Logic via config ---
# Input: data/Extracted_Features
FEATURE_FOLDER = config.FEATURES_DIR

# Output: data/Clustered_BE_datasets/2
OUT_CLUSTER_FOLDER = config.CLUSTERED_DIR / "2"

# Summary: data/clustering_hyperparam_search.csv
SUMMARY_PATH = config.DATA_DIR / "clustering_hyperparam_search.csv"


# ============================================================
# Utility
# ============================================================

def ensure_folder(path: Path):
    os.makedirs(path, exist_ok=True)
    return path


def smooth_labels(df, labels, iterations=3):
    if iterations <= 0:
        return labels
    kx, ky = df["kx"].astype(int).values, df["ky"].astype(int).values
    valid = df["valid"].astype(bool).values
    index_map = {(kx[i], ky[i]): i for i in range(len(df))}
    current = labels.copy()
    for _ in range(iterations):
        new_labels = current.copy()
        for i in range(len(df)):
            if not valid[i] or current[i] == 0:
                continue
            x, y = int(kx[i]), int(ky[i])
            neigh = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    key = (x + dx, y + dy)
                    if key in index_map:
                        j = index_map[key]
                        if valid[j] and current[j] > 0:
                            neigh.append(current[j])
            if neigh:
                new_labels[i] = Counter(neigh).most_common(1)[0][0]
        current = new_labels
    return current


def build_X_for_clustering(df_feat, valid_mask, feature_cols, spatial_weight):
    X_feat = df_feat.loc[valid_mask, feature_cols].values
    X_feat = np.nan_to_num(X_feat, nan=0.0)
    X_feat_scaled = StandardScaler().fit_transform(X_feat)
    XY = df_feat.loc[valid_mask, ["kx", "ky"]].values
    XY_scaled = StandardScaler().fit_transform(XY) * spatial_weight
    return np.hstack([X_feat_scaled, XY_scaled])


def build_X_for_eval(df_with_clusters, feature_cols, spatial_weight):
    mask = (df_with_clusters["valid"] == True) & (df_with_clusters["cluster"] > 0)
    dfv = df_with_clusters.loc[mask].copy()
    if dfv.empty:
        return None, None
    labels = dfv["cluster"].values.astype(int)
    X_feat = np.nan_to_num(dfv[feature_cols].values, nan=0.0)
    X_feat_scaled = StandardScaler().fit_transform(X_feat)
    XY_scaled = StandardScaler().fit_transform(dfv[["kx", "ky"]].values) * spatial_weight
    return np.hstack([X_feat_scaled, XY_scaled]), labels


def feature_metrics(X, labels):
    if X is None or len(np.unique(labels)) < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }
    out = {}
    try:
        out["silhouette"] = silhouette_score(X, labels)
    except Exception:
        out["silhouette"] = np.nan
    try:
        out["davies_bouldin"] = davies_bouldin_score(X, labels)
    except Exception:
        out["davies_bouldin"] = np.nan
    try:
        out["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    except Exception:
        out["calinski_harabasz"] = np.nan
    return out


def spatial_coherence_score(df_with_clusters):
    dfv = df_with_clusters[
        (df_with_clusters["valid"] == True) & (df_with_clusters["cluster"] > 0)
    ].copy()
    if dfv.empty:
        return np.nan
    kx, ky, cl = dfv["kx"].values, dfv["ky"].values, dfv["cluster"].values
    idx = {(kx[i], ky[i]): i for i in range(len(dfv))}
    same, total = 0, 0
    for i in range(len(dfv)):
        x, y = kx[i], ky[i]
        for nx, ny in ((x + 1, y), (x, y + 1)):
            j = idx.get((nx, ny))
            if j is not None:
                total += 1
                if cl[i] == cl[j]:
                    same += 1
    return same / total if total > 0 else np.nan


def run_model(algorithm, X_full, K):
    if algorithm == "kmeans":
        model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
    else:
        model = GaussianMixture(
            n_components=K,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=2,
        )
    return model.fit_predict(X_full) + 1


def is_better(candidate, best):
    if best is None:
        return True
    keys_primary = ["spatial", "silhouette"] if PRIMARY == "spatial" else ["silhouette", "spatial"]
    eps = 1e-8
    for key in keys_primary:
        a, b = candidate.get(key, np.nan), best.get(key, np.nan)
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(b) and not np.isnan(a):
            return True
        if np.isnan(a) and not np.isnan(b):
            return False
        if a > b + eps:
            return True
        if a < b - eps:
            return False
    # DB: smaller is better
    a_db, b_db = candidate.get("davies_bouldin", np.nan), best.get("davies_bouldin", np.nan)
    if not np.isnan(a_db) and not np.isnan(b_db):
        if a_db < b_db - 1e-6:
            return True
    return False


# ============================================================
# Search and Apply
# ============================================================

def search_best_for_file(feature_path):
    df_feat = pd.read_csv(feature_path)
    be_name = os.path.splitext(os.path.basename(feature_path))[0]
    valid_mask = df_feat["valid"].astype(bool).values
    valid_indices = np.where(valid_mask)[0]
    n_valid = valid_mask.sum()
    print(f"\n=== {be_name} === (Valid: {n_valid})")

    best, best_labels = None, None

    for algo in ALGORITHMS:
        for fs_name, feature_cols in FEATURE_SETS.items():
            for K in K_VALUES:
                if n_valid < K:
                    continue
                for w in SPATIAL_WEIGHTS:
                    for sm in SMOOTH_ITERATIONS:
                        X_full = build_X_for_clustering(
                            df_feat, valid_mask, feature_cols, w
                        )
                        labels_valid = run_model(algo, X_full, K)
                        labels_all = np.zeros(len(df_feat), dtype=int)
                        labels_all[valid_indices] = labels_valid

                        df_tmp = df_feat.copy()
                        df_tmp["cluster"] = labels_all
                        labels_sm = smooth_labels(df_tmp, labels_all, iterations=sm)
                        df_tmp["cluster"] = labels_sm

                        X_eval, y_eval = build_X_for_eval(
                            df_tmp, feature_cols, w
                        )
                        fm = feature_metrics(X_eval, y_eval)
                        sc = spatial_coherence_score(df_tmp)

                        candidate = {
                            "algorithm": algo,
                            "feature_set": fs_name,
                            "K": K,
                            "spatial_weight": w,
                            "smooth_iterations": sm,
                            "silhouette": fm["silhouette"],
                            "davies_bouldin": fm["davies_bouldin"],
                            "calinski_harabasz": fm["calinski_harabasz"],
                            "spatial": sc,
                        }

                        if is_better(candidate, best):
                            best, best_labels = candidate, labels_sm.copy()
    return best, best_labels


def apply_best_config(feature_path, best, best_labels):
    df = pd.read_csv(feature_path)
    be_name = os.path.splitext(os.path.basename(feature_path))[0]
    df["cluster"] = best_labels
    out_path = OUT_CLUSTER_FOLDER / f"{be_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


def main():
    if not FEATURE_FOLDER.is_dir():
        raise FileNotFoundError(f"Input folder not found: {FEATURE_FOLDER}")

    ensure_folder(OUT_CLUSTER_FOLDER)
    feature_files = sorted(glob.glob(str(FEATURE_FOLDER / "*.csv")))
    summary_rows = []

    for path in feature_files:
        best, best_labels = search_best_for_file(path)
        if best:
            apply_best_config(path, best, best_labels)
            row = {"BE": os.path.splitext(os.path.basename(path))[0]}
            row.update(best)
            summary_rows.append(row)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
        print(f"\nSummary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

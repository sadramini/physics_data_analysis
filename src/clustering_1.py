import os
import glob
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src import config  # NEW: central config for paths

# ============================================================
# CONFIG – grids to search
# ============================================================

FEATURE_SETS = {
    "A": ["mean_intensity", "std_intensity", "max_intensity", "ts_pca1"],
    "B": ["mean_intensity", "std_intensity", "max_intensity",
          "skew_intensity", "ts_pca1"],
}

SPATIAL_WEIGHTS = [0.0, 0.3, 0.6, 1.0]
SMOOTH_ITERATIONS = [0, 2, 4]

N_CLUSTERS = 3
RANDOM_STATE = 42

# --- Path Setup using config ---
# Input: data/Extracted_Features
FEATURE_FOLDER = config.FEATURES_DIR

# Output Folder: data/Clustered_BE_datasets/1
OUT_CLUSTER_FOLDER = config.CLUSTERED_DIR / "1"

# Summary Path: inside folder '1' to keep results together
SUMMARY_PATH = OUT_CLUSTER_FOLDER / "clustering_hyperparam_search.csv"


# ============================================================
# Utility functions
# ============================================================

def ensure_folder(path):
    # path can be a Path object or string
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def smooth_labels(df, labels, iterations=3):
    """
    Spatial majority smoothing on the kx–ky grid.
    cluster 0 (invalid) is never changed.
    """
    if iterations <= 0:
        return labels

    kx = df["kx"].astype(int).values
    ky = df["ky"].astype(int).values
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
                    key = (x + dx, y + dy)
                    if key in index_map:
                        j = index_map[key]
                        if valid[j] and current[j] > 0:
                            neigh.append(current[j])

            if neigh:
                majority = Counter(neigh).most_common(1)[0][0]
                new_labels[i] = majority

        current = new_labels

    return current


def build_feature_matrix(df, feature_cols, spatial_weight):
    """Return X (features+spatial) and labels (cluster) for valid points."""
    mask = (df["valid"] == True) & (df["cluster"] > 0)
    dfv = df.loc[mask].copy()

    if dfv.empty:
        return None, None

    labels = dfv["cluster"].values.astype(int)

    X_feat = dfv[feature_cols].values
    X_feat = np.nan_to_num(X_feat, nan=0.0)
    feat_scaler = StandardScaler()
    X_feat_scaled = feat_scaler.fit_transform(X_feat)

    kx = dfv["kx"].values.reshape(-1, 1)
    ky = dfv["ky"].values.reshape(-1, 1)
    XY = np.hstack([kx, ky])
    spatial_scaler = StandardScaler()
    XY_scaled = spatial_scaler.fit_transform(XY) * spatial_weight

    X_full = np.hstack([X_feat_scaled, XY_scaled])
    return X_full, labels


def evaluate_clustering(X, labels):
    """Return silhouette, DB, CH metrics for a given embedding + labels."""
    res = {}
    if X is None or len(np.unique(labels)) < 2:
        return {"silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan}

    try:
        res["silhouette"] = silhouette_score(X, labels)
    except Exception:
        res["silhouette"] = np.nan

    try:
        res["davies_bouldin"] = davies_bouldin_score(X, labels)
    except Exception:
        res["davies_bouldin"] = np.nan

    try:
        res["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    except Exception:
        res["calinski_harabasz"] = np.nan

    return res


# ============================================================
# Core search for one BE file
# ============================================================

def search_best_for_file(feature_path):
    """Run the hyperparameter search for a single BE feature file."""
    df_feat = pd.read_csv(feature_path)
    be_name = os.path.splitext(os.path.basename(feature_path))[0]

    if "valid" not in df_feat.columns:
        raise ValueError(f"'valid' column missing in {feature_path}")

    valid_mask = df_feat["valid"].astype(bool).values
    N = len(df_feat)
    valid_indices = np.where(valid_mask)[0]
    n_valid = valid_mask.sum()

    print(f"\n=== {be_name} ===")
    print(f"Valid points: {n_valid}/{N}")

    if n_valid < N_CLUSTERS:
        print("Not enough valid points for clustering, skipping.")
        return None, None

    best_config = None
    best_metrics = None

    for fs_name, feature_cols in FEATURE_SETS.items():
        for spatial_weight in SPATIAL_WEIGHTS:
            for smooth_iter in SMOOTH_ITERATIONS:

                X_feat = df_feat.loc[valid_mask, feature_cols].values
                X_feat = np.nan_to_num(X_feat, nan=0.0)
                scaler = StandardScaler()
                X_feat_scaled = scaler.fit_transform(X_feat)

                kx_valid = df_feat.loc[valid_mask, "kx"].values.reshape(-1, 1)
                ky_valid = df_feat.loc[valid_mask, "ky"].values.reshape(-1, 1)
                XY = np.hstack([kx_valid, ky_valid])
                spatial_scaler = StandardScaler()
                XY_scaled = spatial_scaler.fit_transform(XY) * spatial_weight

                X_full = np.hstack([X_feat_scaled, XY_scaled])

                km = KMeans(
                    n_clusters=N_CLUSTERS,
                    random_state=RANDOM_STATE,
                    n_init="auto",
                )
                labels_valid = km.fit_predict(X_full) + 1  # 1..K

                labels_all = np.zeros(N, dtype=int)
                labels_all[valid_indices] = labels_valid

                df_tmp = df_feat.copy()
                df_tmp["cluster"] = labels_all
                labels_smooth = smooth_labels(df_tmp, labels_all,
                                              iterations=smooth_iter)
                df_tmp["cluster"] = labels_smooth

                X_eval, labels_eval = build_feature_matrix(
                    df_tmp, feature_cols, spatial_weight
                )
                metrics = evaluate_clustering(X_eval, labels_eval)
                sil = metrics["silhouette"]
                db = metrics["davies_bouldin"]
                ch = metrics["calinski_harabasz"]

                if np.isnan(sil):
                    continue

                if best_metrics is None:
                    best_metrics = metrics
                    best_config = {
                        "feature_set": fs_name,
                        "feature_cols": feature_cols,
                        "spatial_weight": spatial_weight,
                        "smooth_iterations": smooth_iter,
                    }
                else:
                    better = False
                    if sil > best_metrics["silhouette"] + 1e-6:
                        better = True
                    elif abs(sil - best_metrics["silhouette"]) <= 1e-6:
                        if db < best_metrics["davies_bouldin"] - 1e-6:
                            better = True
                        elif abs(db - best_metrics["davies_bouldin"]) <= 1e-6:
                            if ch > best_metrics["calinski_harabasz"] + 1e-3:
                                better = True

                    if better:
                        best_metrics = metrics
                        best_config = {
                            "feature_set": fs_name,
                            "feature_cols": feature_cols,
                            "spatial_weight": spatial_weight,
                            "smooth_iterations": smooth_iter,
                        }

    return best_config, best_metrics


def apply_best_config(feature_path, best_config):
    """Re-run clustering with the best config and save to Clustered folder."""
    df = pd.read_csv(feature_path)
    be_name = os.path.splitext(os.path.basename(feature_path))[0]

    feature_cols = best_config["feature_cols"]
    spatial_weight = best_config["spatial_weight"]
    smooth_iter = best_config["smooth_iterations"]

    valid_mask = df["valid"].astype(bool).values
    N = len(df)
    valid_indices = np.where(valid_mask)[0]

    X_feat = df.loc[valid_mask, feature_cols].values
    X_feat = np.nan_to_num(X_feat, nan=0.0)
    scaler = StandardScaler()
    X_feat_scaled = scaler.fit_transform(X_feat)

    kx_valid = df.loc[valid_mask, "kx"].values.reshape(-1, 1)
    ky_valid = df.loc[valid_mask, "ky"].values.reshape(-1, 1)
    XY = np.hstack([kx_valid, ky_valid])
    spatial_scaler = StandardScaler()
    XY_scaled = spatial_scaler.fit_transform(XY) * spatial_weight

    X_full = np.hstack([X_feat_scaled, XY_scaled])

    km = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init="auto",
    )
    labels_valid = km.fit_predict(X_full) + 1

    labels_all = np.zeros(N, dtype=int)
    labels_all[valid_indices] = labels_valid

    df["cluster"] = labels_all
    labels_smooth = smooth_labels(df, labels_all, iterations=smooth_iter)
    df["cluster"] = labels_smooth

    # Ensure the '1' folder exists
    ensure_folder(OUT_CLUSTER_FOLDER)
    out_path = OUT_CLUSTER_FOLDER / f"{be_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved final clustered file -> {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    feature_files = sorted(glob.glob(os.path.join(str(FEATURE_FOLDER), "*.csv")))
    if not feature_files:
        raise FileNotFoundError(f"No feature files found in '{FEATURE_FOLDER}'")

    summary_rows = []

    for path in feature_files:
        be_name = os.path.splitext(os.path.basename(path))[0]
        best_cfg, best_met = search_best_for_file(path)
        if best_cfg is None:
            continue

        apply_best_config(path, best_cfg)

        row = {
            "BE": be_name,
            "feature_set": best_cfg["feature_set"],
            "spatial_weight": best_cfg["spatial_weight"],
            "smooth_iterations": best_cfg["smooth_iterations"],
            "silhouette": best_met["silhouette"],
            "davies_bouldin": best_met["davies_bouldin"],
            "calinski_harabasz": best_met["calinski_harabasz"],
        }
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        # Ensure the summary folder exists
        ensure_folder(OUT_CLUSTER_FOLDER)
        summary_df.to_csv(SUMMARY_PATH, index=False)
        print("\nHyperparameter search summary saved to", SUMMARY_PATH)


if __name__ == "__main__":
    main()

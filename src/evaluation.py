import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src import config  # NEW: central config for paths

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
VERSION = 1  # Matches the folder number in Clustered_BE_datasets
SPATIAL_WEIGHT = 0.5  # Should match what was used for this cluster set

FEATURE_COLS = [
    "mean_intensity",
    "std_intensity",
    "max_intensity",
    "skew_intensity",
    "ts_pca1",
]

# -------------------------------------------------------------------
# Path Logic (via config)
# -------------------------------------------------------------------

# Input: data/Clustered_BE_datasets/{VERSION}
CLUSTERED_FOLDER = config.CLUSTERED_DIR / str(VERSION)

# Output: results/evaluation/{VERSION}
OUTPUT_FOLDER = config.RESULTS_DIR / "evaluation" / str(VERSION)


def ensure_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_feature_matrix(df, spatial_weight=0.5):
    """
    Rebuilds the same feature matrix used in clustering for evaluation.
    """
    mask = (df["valid"] == True) & (df["cluster"] > 0)
    dfv = df.loc[mask].copy()

    if dfv.empty:
        return None, None

    labels = dfv["cluster"].values.astype(int)

    X_feat = dfv[FEATURE_COLS].values
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


def evaluate_one_file(path):
    # Skip the summary file if it exists in the folder
    if "clustering_hyperparam_search" in str(path):
        return None

    df = pd.read_csv(path)

    # Check if necessary columns exist (avoiding evaluation on summary files)
    if "kx" not in df.columns or "cluster" not in df.columns:
        return None

    X, labels = build_feature_matrix(df, spatial_weight=SPATIAL_WEIGHT)

    if X is None or len(np.unique(labels)) < 2:
        print(f"{os.path.basename(path)} -> not enough labeled points for metrics.")
        return None

    results = {}
    try:
        results["silhouette"] = silhouette_score(X, labels)
    except Exception as e:
        results["silhouette"] = np.nan

    try:
        results["davies_bouldin"] = davies_bouldin_score(X, labels)
    except Exception as e:
        results["davies_bouldin"] = np.nan

    try:
        results["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    except Exception as e:
        results["calinski_harabasz"] = np.nan

    return results


def main():
    if not CLUSTERED_FOLDER.is_dir():
        raise FileNotFoundError(f"Clustered folder not found: {CLUSTERED_FOLDER}")

    ensure_folder(OUTPUT_FOLDER)

    files = sorted(glob.glob(os.path.join(str(CLUSTERED_FOLDER), "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSVs found in '{CLUSTERED_FOLDER}'")

    print(f"Evaluating Set {VERSION} ({len(files)} files)...\n")

    rows = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]

        # Skip summary CSVs
        if "clustering_hyperparam_search" in name:
            continue

        res = evaluate_one_file(path)
        if res is not None:
            print(f"File: {name}")
            row = {"BE": name}
            row.update(res)
            rows.append(row)
            print(
                f"  Sil: {res['silhouette']:.4f} | "
                f"DB: {res['davies_bouldin']:.4f} | "
                f"CH: {res['calinski_harabasz']:.2f}"
            )
            print("-" * 50)

    if rows:
        summary = pd.DataFrame(rows).set_index("BE")
        summary_path = OUTPUT_FOLDER / "clustering_metrics_summary.csv"
        summary.to_csv(summary_path)

        print("\n" + "=" * 30)
        print(f"EVALUATION SUMMARY (Set {VERSION})")
        print("=" * 30)
        print(summary)
        print(f"\nFinal metrics saved to: {summary_path}")


if __name__ == "__main__":
    main()

import os
import glob
import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src import config  # NEW: central config for paths


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def ensure_folder(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def extract_be_value_from_filename(filename: str) -> str:
    """
    Extract BE value from: BE_<value>_preprocessed.csv -> <value>
    """
    base = os.path.basename(filename)
    m = re.match(r"BE_(.+?)_preprocessed\.csv", base)
    if not m:
        raise ValueError(f"Cannot extract BE value from filename: {base}")
    return m.group(1)


def row_moments(ts: np.ndarray):
    """
    Compute mean, std, skewness for a 1D array (population formulas).
    """
    ts = np.asarray(ts, dtype=float)
    n = ts.size
    if n == 0:
        return np.nan, np.nan, np.nan

    mean = ts.mean()
    diff = ts - mean
    var = (diff ** 2).mean()
    std = np.sqrt(var)

    if std < 1e-12:
        skew = 0.0
    else:
        z = diff / std
        skew = (z ** 3).mean()

    return float(mean), float(std), float(skew)


# ---------------------------------------------------------
# Per-BE processing
# ---------------------------------------------------------

def process_be_file(preproc_path: str,
                    out_folder: str):
    """
    For a single BE preprocessed CSV:
      - compute amplitude stats (mean, std, max, skew) on t1..tN
      - run PCA (PC1 only) on full time-series (valid rows only)
      - save features to Extracted Features/<BE>.csv
    """
    print(f"\nProcessing: {preproc_path}")
    df = pd.read_csv(preproc_path)

    if "kx" not in df.columns or "ky" not in df.columns or "valid" not in df.columns:
        raise ValueError("Preprocessed CSV must contain 'kx', 'ky', and 'valid' columns.")

    # time-series columns
    t_cols = [c for c in df.columns if c.startswith("t")]
    # ensure t1..t51 are in order
    t_cols = sorted(t_cols, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    ts_all = df[t_cols].values
    valid_mask = df["valid"].astype(bool).values

    n_rows, n_time = ts_all.shape
    print(f"  Rows: {n_rows}, time points: {n_time}, valid rows: {valid_mask.sum()}")

    # ---------------------------
    # amplitude stats per row
    # ---------------------------
    mean_arr = np.full(n_rows, np.nan, dtype=float)
    std_arr = np.full(n_rows, np.nan, dtype=float)
    max_arr = np.full(n_rows, np.nan, dtype=float)
    skew_arr = np.full(n_rows, np.nan, dtype=float)

    for i in range(n_rows):
        if not valid_mask[i]:
            continue
        ts = ts_all[i, :]
        m, s, sk = row_moments(ts)
        mean_arr[i] = m
        std_arr[i] = s
        max_arr[i] = float(np.max(ts))
        skew_arr[i] = sk

    # ---------------------------
    # PCA on full time-series (valid only) → keep PC1
    # ---------------------------
    ts_valid = ts_all[valid_mask]
    if ts_valid.shape[0] == 0:
        print("  No valid rows; PCA component will be NaN.")
        ts_pca1_full = np.full(n_rows, np.nan, dtype=float)
    else:
        scaler = StandardScaler()
        ts_valid_scaled = scaler.fit_transform(ts_valid)

        pca = PCA(n_components=1)
        ts_valid_pcs = pca.fit_transform(ts_valid_scaled)  # shape (n_valid, 1)
        print("  Time-series PCA EVR (PC1):", pca.explained_variance_ratio_[0])

        ts_pca1_full = np.full(n_rows, np.nan, dtype=float)
        valid_indices = np.where(valid_mask)[0]
        for j, idx in enumerate(valid_indices):
            ts_pca1_full[idx] = ts_valid_pcs[j, 0]

    # ---------------------------
    # build output DataFrame
    # ---------------------------
    out_df = pd.DataFrame({
        "kx": df["kx"].values.astype(float),
        "ky": df["ky"].values.astype(float),
        "valid": valid_mask.astype(bool),
        "mean_intensity": mean_arr,
        "std_intensity": std_arr,
        "max_intensity": max_arr,
        "skew_intensity": skew_arr,
        "ts_pca1": ts_pca1_full,
    })

    be_value = extract_be_value_from_filename(preproc_path)
    out_path = os.path.join(out_folder, f"{be_value}.csv")
    out_df.to_csv(out_path, index=False)

    print(f"  Saved features → {out_path}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    # input folder: data/Preprocessed_BE_datasets (from config)
    preproc_folder = config.PREPROCESSED_DIR

    # output folder: data/Extracted_Features (from config)
    out_folder = config.FEATURES_DIR

    if not preproc_folder.is_dir():
        raise FileNotFoundError(f"Preprocessed folder not found: {preproc_folder}")

    # Ensure output folder exists in data/
    ensure_folder(str(out_folder))

    # Search for files in the preprocessed folder
    files = sorted(glob.glob(str(preproc_folder / "BE_*_preprocessed.csv")))
    if not files:
        raise FileNotFoundError(f"No preprocessed BE CSVs found in {preproc_folder}")

    print("Found preprocessed BE datasets:")
    for f in files:
        print(" ", f)

    for f in files:
        process_be_file(f, str(out_folder))

    print("\nAll BE feature files created.")


if __name__ == "__main__":
    main()

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from pathlib import Path
from src import config  # NEW: central config for paths


# ---------------------------------------------------------
# Improved preprocessing for VG (stronger per-row smoothing)
# ---------------------------------------------------------

def robust_baseline_row(ts, frac=0.20):
    """Baseline = median of lowest frac of values (row-wise)."""
    ts = np.asarray(ts, dtype=float)
    ts_sorted = np.sort(ts)
    cut = max(1, int(len(ts) * frac))
    return np.median(ts_sorted[:cut])


def moving_average(x, window=3):
    """Simple centered moving average with edge padding."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x
    pad = window // 2
    x_pad = np.pad(x, pad_width=pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(x_pad, kernel, mode="valid")


def preprocess_be_timeseries(ts_matrix,
                             savgol_window=13,
                             savgol_polyorder=2,
                             ma_window=3,
                             global_frac_for_norm=99,
                             min_relative_peak=0.05):
    """
    Preprocess a BE's time-series matrix (num_rows x num_t):

      1. Robust baseline subtraction per row.
      2. Strong Savgol smoothing per row.
      3. Moving-average smoothing per row.
      4. Global normalization per BE (percentile scaling).
      5. Valid mask based on peak amplitude.
    """
    ts_matrix = np.asarray(ts_matrix, dtype=float)
    num_rows, num_t = ts_matrix.shape

    # 1) Baseline subtraction
    baseline_sub = np.zeros_like(ts_matrix)
    for i in range(num_rows):
        row = ts_matrix[i, :]
        b = robust_baseline_row(row)
        r = row - b
        r[r < 0] = 0.0
        baseline_sub[i, :] = r

    # 2) Savitzky–Golay smoothing (stronger: larger window)
    wl = min(savgol_window, num_t if num_t % 2 == 1 else num_t - 1)
    wl = max(wl, 5)
    if wl % 2 == 0:
        wl += 1

    smoothed = np.zeros_like(baseline_sub)
    for i in range(num_rows):
        row = baseline_sub[i, :]
        sm = savgol_filter(row, window_length=wl,
                           polyorder=savgol_polyorder,
                           mode="interp")
        sm[sm < 0] = 0.0
        smoothed[i, :] = sm

    # 3) Moving-average smoothing (extra denoising)
    further_smoothed = np.zeros_like(smoothed)
    for i in range(num_rows):
        further_smoothed[i, :] = moving_average(smoothed[i, :], window=ma_window)
        further_smoothed[i, further_smoothed[i, :] < 0] = 0.0

    # 4) Global normalization (per BE, not per pixel)
    flat = further_smoothed.flatten()
    global_scale = np.percentile(flat, global_frac_for_norm)

    if global_scale <= 0:
        return np.zeros_like(further_smoothed), np.zeros(num_rows, dtype=bool)

    ts_out = further_smoothed / global_scale
    ts_out = np.clip(ts_out, 0.0, 1.5)

    # 5) Valid mask: row is valid if its max >= min_relative_peak
    row_max = np.max(ts_out, axis=1)
    valid = row_max >= min_relative_peak

    return ts_out, valid


# ---------------------------------------------------------
# File-level processing
# ---------------------------------------------------------

def process_be_file(input_path, output_folder):
    """
    EXACT same output naming as before:
      Preprocessed_BE_datasets/BE_<value>_preprocessed.csv
    """
    print(f"\nProcessing raw BE dataset for VG: {input_path}")
    df = pd.read_csv(input_path)

    # Identify time columns (everything except kx, ky)
    non_time = {"kx", "ky"}
    time_cols = [c for c in df.columns if c not in non_time]

    # Sort time columns like t1, t2, ..., t51
    time_cols = sorted(
        time_cols,
        key=lambda x: int(x[1:]) if x.startswith("t") and x[1:].isdigit() else x
    )

    ts_matrix = df[time_cols].values
    num_rows, num_t = ts_matrix.shape
    print(f"  → {num_rows} rows, {num_t} time points")

    # Apply improved preprocessing
    ts_out, valid = preprocess_be_timeseries(ts_matrix)

    # Write back into DataFrame
    out_df = df.copy()
    out_df[time_cols] = ts_out
    out_df["valid"] = valid

    # Output naming EXACTLY like before
    base = os.path.basename(input_path)           # e.g. BE_2.0999999.csv
    root, _ = os.path.splitext(base)              # BE_2.0999999
    out_name = f"{root}_preprocessed.csv"         # BE_2.0999999_preprocessed.csv
    out_path = output_folder / out_name           # use Path join

    out_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


def main():
    # Input folder: data/raw/BE_datasets (from config)
    input_folder = config.RAW_BE_DIR

    # Output folder: data/Preprocessed_BE_datasets (from config)
    output_folder = config.PREPROCESSED_DIR

    if not input_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)
    print("Preprocessed output will be saved in:", output_folder)

    # Use glob on the specific input path
    be_files = sorted(glob.glob(str(input_folder / "BE_*.csv")))
    if not be_files:
        raise FileNotFoundError(f"No BE_*.csv files found in {input_folder}")

    for f in be_files:
        process_be_file(f, output_folder)

    print("\nAll BE datasets preprocessed (stronger smoothing per time series).")


if __name__ == "__main__":
    main()

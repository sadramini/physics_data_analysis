import numpy as np
import pandas as pd
from pathlib import Path
from src import config  # NEW: use central config for paths

# --------- 1) Helpers to read numeric blocks from ITX ---------

def load_itx_block(path):
    """
    Read the numeric data between BEGIN/END in an ITX file.
    Returns a 2D numpy array: rows = lines, columns = numbers per line.
    """
    lines = Path(path).read_text().splitlines()

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if "BEGIN" in line and start_idx is None:
            start_idx = i + 1
        elif "END" in line and start_idx is not None:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise ValueError(f"BEGIN/END markers not found correctly in {path}")

    data_rows = []
    for line in lines[start_idx:end_idx]:
        parts = line.strip().split()
        if parts:
            data_rows.append([float(x) for x in parts])

    return np.array(data_rows, dtype=float)


def load_itx_1d(path):
    """
    Load a 1D wave (BE or Time Steps) from ITX.
    """
    arr2d = load_itx_block(path)
    return arr2d.ravel()


# --------- 2) Load the three ITX files ---------

# Use ITX directory from config (.env-controlled)
itx_dir = config.ITX_DIR

# Loading files from the itx folder
data_2d   = load_itx_block(itx_dir / "4D Datastack.itx")   # (51000, 100)
BE_vals   = load_itx_1d(itx_dir / "BE.itx")                # (10,)
time_vals = load_itx_1d(itx_dir / "Time Steps.itx")        # (51,)

print("4D Datastack block shape:", data_2d.shape)
print("BE values:", BE_vals)
print("Time steps shape:", time_vals.shape)

# Expected sizes
n_kx   = 100
n_ky   = 100
n_time = 51
n_be   = 10

expected_rows = n_be * n_time * n_ky    # 10 * 51 * 100 = 51000
expected_cols = n_kx                    # 100

if data_2d.shape != (expected_rows, expected_cols):
    raise ValueError(
        f"Unexpected shape for 4D Datastack: {data_2d.shape}, "
        f"expected {(expected_rows, expected_cols)}"
    )
if BE_vals.size != n_be:
    raise ValueError(f"Expected {n_be} BE values, got {BE_vals.size}")
if time_vals.size != n_time:
    raise ValueError(f"Expected {n_time} time steps, got {time_vals.size}")


# --------- 3) Reshape into 4D: (BE, time, ky, kx) ---------
# Igor ASCII export is effectively loops over:
#   for be in BE:
#     for t in time:
#       for ky in 0..99:
#         print 100 kx-values
#
# So the natural 4D layout is (n_be, n_time, n_ky, n_kx)

cube_be_t_ky_kx = data_2d.reshape((n_be, n_time, n_ky, n_kx))

# Reorder to (kx, ky, BE, time) for convenience:
# axes: (be, time, ky, kx) -> (kx, ky, be, time)
cube_kx_ky_be_t = np.transpose(cube_be_t_ky_kx, (3, 2, 0, 1))
# shape: (n_kx, n_ky, n_be, n_time) = (100, 100, 10, 51)

print("4D cube shape (kx, ky, BE, time):", cube_kx_ky_be_t.shape)


# --------- 4) Flatten to a table: one row = (kx, ky, BE) ---------

# Create index grids aligned with cube_kx_ky_be_t
kx_idx, ky_idx, be_idx = np.meshgrid(
    np.arange(n_kx), np.arange(n_ky), np.arange(n_be),
    indexing="ij"
)

# Flatten indices
kx_vals_flat = (kx_idx.ravel() + 1)           # 1..100
ky_vals_flat = (ky_idx.ravel() + 1)           # 1..100
BE_flat      = BE_vals[be_idx.ravel()]       # map indices -> actual BE

# Flatten intensities: (n_kx * n_ky * n_be, n_time)
time_series = cube_kx_ky_be_t.reshape(-1, n_time)

# Build DataFrame
time_cols = [f"t{i+1}" for i in range(n_time)]
df = pd.DataFrame(
    np.column_stack([kx_vals_flat, ky_vals_flat, BE_flat, time_series]),
    columns=["kx", "ky", "BE"] + time_cols
)

print("Final dataframe shape:", df.shape)
print(df.head())

# --------- 5) Save to CSV ---------

# Output directory: use DATA_DIR/raw from config
output_dir = config.DATA_DIR / "raw"

# Create the directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Save the file
output_path = output_dir / "reshaped_dataset.csv"
df.to_csv(output_path, index=False)

print(f"Saved CSV as '{output_path}'")

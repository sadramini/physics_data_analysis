import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path

from src import config  # NEW: central config paths

# ============================================================
# CONFIG
# ============================================================
VERSION = 1


def ensure_folder(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_be_clusters(csv_path: str, output_folder: Path, version_num: int) -> None:
    # --- Skip the Summary File ---
    if "clustering_hyperparam_search" in csv_path:
        return

    print(f"\nVisualizing clusters for: {csv_path}")
    df = pd.read_csv(csv_path)

    if "kx" not in df.columns or "ky" not in df.columns or "cluster" not in df.columns:
        raise ValueError(
            f"Input CSV {os.path.basename(csv_path)} must contain 'kx', 'ky', and 'cluster' columns."
        )

    kx = df["kx"].values
    ky = df["ky"].values
    clusters = df["cluster"].values.astype(int)

    unique_clusters = np.unique(clusters)
    unique_clusters_sorted = np.sort(unique_clusters)

    num_real_clusters = np.count_nonzero(unique_clusters_sorted != 0)
    base_cmap = plt.cm.get_cmap("tab20", max(num_real_clusters, 1))

    colors = ["lightgrey"]
    if num_real_clusters > 0:
        colors.extend(base_cmap.colors[:num_real_clusters])

    cmap = ListedColormap(colors)

    nonzero_clusters = [c for c in unique_clusters_sorted if c != 0]
    cluster_to_idx = {0: 0}
    for i, c in enumerate(nonzero_clusters, start=1):
        cluster_to_idx[c] = i

    color_indices = np.array([cluster_to_idx[c] for c in clusters])

    plt.figure(figsize=(6, 6))
    plt.scatter(
        kx,
        ky,
        c=color_indices,
        cmap=cmap,
        s=20,
        marker="s",
        edgecolors="none",
    )

    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.title(
        f"{os.path.splitext(os.path.basename(csv_path))[0]} (Set {version_num})"
    )
    plt.gca().set_aspect("equal", adjustable="box")

    legend_handles = []
    for c in unique_clusters_sorted:
        idx = cluster_to_idx[c]
        color = cmap(idx)
        label = "Cluster 0 (noise)" if c == 0 else f"Cluster {c}"
        legend_handles.append(Patch(facecolor=color, edgecolor="none", label=label))

    plt.legend(handles=legend_handles, loc="best", fontsize=8)
    plt.tight_layout()

    be_name = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = output_folder / f"{be_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved plot → {out_path}")


def main():
    # Input clustered dataset folder (config + VERSION)
    clustered_folder = config.CLUSTERED_DIR / str(VERSION)

    # Output plots folder under results/
    output_folder = config.RESULTS_DIR / "Cluster_Plots" / str(VERSION)

    if not clustered_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {clustered_folder}")

    ensure_folder(output_folder)

    # Get all CSVs
    csv_files = sorted(glob.glob(os.path.join(str(clustered_folder), "*.csv")))

    for f in csv_files:
        # Skip summary file
        if "clustering_hyperparam_search" in f:
            print(f"--- Skipping summary file: {os.path.basename(f)} ---")
            continue

        plot_be_clusters(f, output_folder, VERSION)

    print(f"\nAll cluster maps for set {VERSION} created successfully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Run the full Brillouin Energy analysis pipeline end-to-end.

Uses environment-based configuration from:
  - .env (user local settings)
  - .env.example (template)

Steps:
1. Convert .itx -> CSV
2. Split dataset into BE_xx files
3. Preprocess BE datasets
4. Extract features
5. Run clustering method 1
6. Run clustering method 2
7. Generate visualizations
8. Evaluate clustering outputs
"""

from src import config
from src.io import itx_to_csv   # runs conversion on import (for now)
from src.io.split_dataset import main as split_be_main
from src import (
    preprocessing,
    features,
    clustering_1,
    clustering_2,
    visualization,
    evaluation,
)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_paths():
    print_header("ACTIVE PIPELINE PATHS (from .env)")
    print("DATA_DIR:               ", config.DATA_DIR)
    print("RESULTS_DIR:            ", config.RESULTS_DIR)
    print("ITX_DIR:                ", config.ITX_DIR)
    print("RAW_BE_DIR:             ", config.RAW_BE_DIR)
    print("PREPROCESSED_DIR:       ", config.PREPROCESSED_DIR)
    print("FEATURES_DIR:           ", config.FEATURES_DIR)
    print("CLUSTERED_DIR:          ", config.CLUSTERED_DIR)
    print("\n(override any of these in your .env file)")


def main():

    show_paths()

    print_header("STEP 1 — Convert ITX -> CSV")
    print("Source:", config.ITX_DIR)
    print("Output: reshaped CSV in data/raw/")
    # NOTE: itx_to_csv executes on import — no function call yet

    print_header("STEP 2 — Split reshaped dataset -> BE_xx files")
    split_be_main()

    print_header("STEP 3 — Preprocess BE datasets")
    preprocessing.main()

    print_header("STEP 4 — Extract features")
    features.main()

    print_header("STEP 5 — Clustering (Method 1)")
    clustering_1.main()

    #print_header("STEP 6 — Clustering (Method 2)")
    #clustering_2.main()

    print_header("STEP 7 — Visualization")
    visualization.main()

    print_header("STEP 8 — Evaluation")
    evaluation.main()

    print("\nPipeline finished successfully ✅")


if __name__ == "__main__":
    main()

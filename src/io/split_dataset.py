import os
import pandas as pd
from src import config  # NEW: use central config paths


def main():
    # --- Path logic: use config instead of manual joins ---
    # Raw data directory (e.g. ./data/raw)
    raw_path = config.DATA_DIR / "raw"

    # Path to input file (pointing to data/raw)
    input_path = raw_path / "reshaped_dataset.csv"

    print("Loading dataset:", input_path)
    df = pd.read_csv(input_path)

    # Make sure we have needed columns
    if not {"kx", "ky", "BE"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'kx', 'ky', and 'BE' columns.")

    # Output folder for BE-split datasets (e.g. ./data/raw/BE_datasets or from .env)
    output_folder = config.RAW_BE_DIR
    os.makedirs(output_folder, exist_ok=True)
    print("Output folder:", output_folder)

    # Find all unique BE values
    be_values = sorted(df["BE"].unique())
    print(f"Found {len(be_values)} BE values:", be_values)

    # Generate files
    for be in be_values:
        be_df = df[df["BE"] == be].copy()

        # Remove the BE column
        be_df = be_df.drop(columns=["BE"])

        # Reset index
        be_df = be_df.reset_index(drop=True)

        # Save
        filename = f"BE_{be}.csv"
        filepath = output_folder / filename

        be_df.to_csv(filepath, index=False)
        print(f"Saved dataset for BE = {be} → {filepath}, rows = {len(be_df)}")

    print("\nDone splitting BE datasets!")


if __name__ == "__main__":
    main()

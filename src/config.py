from pathlib import Path
import os
from dotenv import load_dotenv

# Base directory of the project (repo root)
BASE_DIR = Path(__file__).resolve().parents[1]

# Load .env from project root if it exists
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Helper to read env with a default
def _get_path(var_name: str, default_relative: str) -> Path:
    value = os.getenv(var_name, default_relative)
    return (BASE_DIR / value).resolve()

DATA_DIR = _get_path("DATA_DIR", "data")
RESULTS_DIR = _get_path("RESULTS_DIR", "results")

ITX_DIR = _get_path("ITX_DIR", "data/itx")
RAW_BE_DIR = _get_path("RAW_BE_DIR", "data/raw/BE_datasets")
PREPROCESSED_DIR = _get_path("PREPROCESSED_DIR", "data/Preprocessed_BE_datasets")
FEATURES_DIR = _get_path("FEATURES_DIR", "data/Extracted_Features")
CLUSTERED_DIR = _get_path("CLUSTERED_DIR", "data/Clustered_BE_datasets")

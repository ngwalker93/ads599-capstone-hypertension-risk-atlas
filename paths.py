from pathlib import Path

def get_project_root():
    current_path = Path.cwd()
    while current_path.name != "ADS599_Capstone_Hypertension_Risk_Atlas":
        if current_path == current_path.parent:
            raise FileNotFoundError("Could not find project root!")
        current_path = current_path.parent
    return current_path

BASE_PATH = get_project_root()
DATA_RAW = BASE_PATH / "data" / "raw"
DATA_PROCESSED = BASE_PATH / "data" / "processed"
FIGURES_DIR = BASE_PATH / "data" / "figures"
VALIDATION_DIR = BASE_PATH / "data" / "validation"
from pathlib import Path
import sys

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
VALIDATION_DIR = BASE_PATH / "data" / "validation"
DATA_FINAL = BASE_PATH / "data" / "final"

def validate_and_alert(file_path, script_name, instructions):
    """
    Checks if a file exists. If not, prints specific instructions and exits.
    """
    if not Path(file_path).exists():
        print("!" * 60)
        print(f"CRITICAL: Missing file for {script_name}!")
        print(f"Expected path: {file_path}")
        print("-" * 60)
        print("INSTRUCTIONS TO RETRIEVE DATA:")
        print(instructions)
        print("!" * 60)
        sys.exit(1)
    else:
        print(f"✅ Data found for {script_name}. Proceeding...")
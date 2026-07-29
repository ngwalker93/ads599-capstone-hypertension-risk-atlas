"""
This script runs the entire data ingestion pipeline by executing each step in sequence.
It ensures that each step is executed in the correct order and handles any errors that may arise during execution.
"""
import subprocess
import sys
import os
from pathlib import Path

# Define the variable at the global level so all functions can see it
PIPELINE_STEPS = [
    "pipeline/02_DataIngestion/01_process_cdc.py",
    "pipeline/02_DataIngestion/02_process_usda.py",
    "pipeline/02_DataIngestion/03_process_chrr_census.py",
    "pipeline/02_DataIngestion/04_process_census_acs.py",
    "pipeline/02_DataIngestion/05_master_dataset.py",
    "pipeline/02_DataIngestion/06_load_to_sql.py"
]

# Get the project root directory (the folder above pipeline/DataIngestion)
project_root = str(Path(__file__).resolve().parent.parent.parent)

def run_pipeline():
    # Create a copy of the current environment variables
    env = os.environ.copy()
    # Add the project root to PYTHONPATH
    env["PYTHONPATH"] = project_root
    
    print(f"🚀 Starting Full Pipeline Execution with root: {project_root}...")

    for script in PIPELINE_STEPS:
        print(f"\n--- Executing: {script} ---")
        try:
            # Pass the modified 'env' to the subprocess
            subprocess.run([sys.executable, script], check=True, env=env)
            print(f"✅ Finished: {script}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Pipeline failed at {script}")
            sys.exit(1)

    print("------------------------------------------------------------------")
    print("🚀 DataIngestion completed successfully. Moving to Feature Engineering")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    run_pipeline()
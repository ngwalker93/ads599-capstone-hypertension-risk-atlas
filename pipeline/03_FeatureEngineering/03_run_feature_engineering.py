"""
This script runs the entire feature engineering pipeline by executing each step in sequence.
It ensures that each step is executed in the correct order and handles any errors that may arise during execution.
"""
import subprocess
import sys
import os
from pathlib import Path

# Define the variable at the global level so all functions can see it
PIPELINE_STEPS = [
    "pipeline/03_FeatureEngineering/01_feature_engineering.py",
    "pipeline/03_FeatureEngineering/02_train_test_split.py"
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
    print("🚀 Feature Engineering completed successfully. Moving to Pipeline Reports...")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    run_pipeline()
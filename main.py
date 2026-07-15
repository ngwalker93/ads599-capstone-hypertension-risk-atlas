"""
This script serves as the main entry point for the entire project pipeline. 
It sequentially executes all the necessary scripts in the DataIngestion pipeline, 
ensuring that each step is completed successfully before moving on to the next. If any script fails, 
the pipeline halts and reports the error, preventing further execution and allowing for debugging.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_full_project():
    print("🚀 Launching Full Project Pipeline...")
    
    # Ensure the environment knows where the root is
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    
    # Use sys.executable to ensure we use the same environment
    # Using 'check=True' will raise an error if these fail, 
    # making it impossible to "silently" fail.
    try:
        subprocess.run([sys.executable, "pipeline/DataIngestion/07_run_ingestion.py"], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline halted due to error: {e}")
        sys.exit(1)
    
    print("✅ Full Project Complete!")

if __name__ == "__main__":
    run_full_project()
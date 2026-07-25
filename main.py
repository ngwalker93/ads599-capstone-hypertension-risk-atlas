"""
This script serves as the main entry point for the entire project pipeline. 
It sequentially executes all necessary pipeline stages, ensuring that each step 
is completed successfully before moving on to the next.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_full_project():
    print("🚀 Launching Full Project Pipeline...")
    
    # Get the actual top-level project root directory 
    project_root = Path(__file__).resolve().parent
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    # Define your sequential master scripts in the correct execution order
    pipeline_stages = [
        "pipeline/02_DataIngestion/07_run_ingestion.py",
        "pipeline/03_FeatureEngineering/03_run_feature_engineering.py",
        "pipeline/04_generate_report.py"
    ]
    
    try:
        for script in pipeline_stages:
            print(f"\n--- Executing Stage: {script} ---")
            subprocess.run([sys.executable, script], check=True, env=env)
            print(f"✅ Completed: {script}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline halted due to error in script: {e}")
        sys.exit(1)
    
    print("\n✅ Full Project Complete!")

if __name__ == "__main__":
    run_full_project()
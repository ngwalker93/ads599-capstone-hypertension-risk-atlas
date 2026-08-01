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
    
    # Define your sequential master scripts in the correct execution order using Path objects
    pipeline_stages = [
        project_root / "pipeline" / "02_DataIngestion" / "07_run_ingestion.py",
        project_root / "pipeline" / "03_FeatureEngineering" / "03_run_feature_engineering.py",
        project_root / "pipeline" / "04_generate_eda_report.py",
        project_root / "pipeline" / "05_generate_modeling_report.py"
    ]
    
    try:
        for script_path in pipeline_stages:
            print(f"\n--- Executing Stage: {script_path.relative_to(project_root)} ---")
            
            # Stream output in real-time instead of buffering
            process = subprocess.Popen(
                [sys.executable, str(script_path)], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                env=env, 
                text=True, 
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end="")
                
            process.wait()
            
            if process.returncode != 0:
                print(f"❌ Pipeline halted in script: {script_path.name}")
                sys.exit(1)
                
            print(f"✅ Completed: {script_path.name}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline halted in script: {script_path.name}")
        print(f"--- STDOUT ---\n{e.stdout}")
        print(f"--- STDERR ---\n{e.stderr}")
        sys.exit(1)
    
    print("\n✅ Full Project Complete!")

if __name__ == "__main__":
    run_full_project()
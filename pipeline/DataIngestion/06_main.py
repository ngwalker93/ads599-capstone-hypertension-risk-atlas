import subprocess
import sys
from pathlib import Path

import os

def check_raw_data_exists():
    raw_dir = Path.cwd() / "data" / "raw"
    
    # Map filenames to their download instructions
    requirements = {
        "cdc_places_raw.csv": (
            "CDC PLACES Data",
            "https://data.cdc.gov/resource/cwsq-ngmh.csv"
        ),
        "FoodAccessResearchAtlasData2019.xlsx": (
            "USDA Food Atlas",
            "https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data"
        ),
        "analytic_data2025.csv": (
            "CHR&R Analytic Data",
            "https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation"
        )
    }
    
    # Check for missing files
    missing = [f for f in requirements if not (raw_dir / f).exists()]
    
    # Check for Census API Key (Required for the final dataset)
    missing_api_key = "CENSUS_API_KEY" not in os.environ
    
    if missing or missing_api_key:
        print("❌ PIPELINE PRE-FLIGHT CHECK FAILED")
        for f in missing:
            name, url = requirements[f]
            print(f"\n- Missing: {name}")
            print(f"  Download from: {url}")
            print(f"  Move and rename to: data/raw/{f}")
        
        if missing_api_key:
            print("\n- Missing: Census API Key")
            print("  Please get a key here: https://api.census.gov/data/key_signup.html")
            print("  Then set it in your terminal: export CENSUS_API_KEY='your_key_here'")
        sys.exit(1)

def run_all():
    scripts = [
        "scripts/01_process_cdc.py", 
        "scripts/02_process_usda.py", 
        "scripts/03_process_chrr_census.py",
        "scripts/04_master_merge.py"
    ]
    
    for script in scripts:
        print(f"🚀 Running {script}...")
        subprocess.run([sys.executable, script], check=True) # Use sys.executable for better compatibility
    
    print("✅ Pipeline complete! Master dataset is in data/processed/")

if __name__ == "__main__":
    check_raw_data_exists()
    run_all()
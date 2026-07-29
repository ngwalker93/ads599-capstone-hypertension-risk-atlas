"""
This script processes the Census ACS data for counties in the U.S. for the year 2023. 
It checks for the existence of the raw data file, cleans it by handling missing values and
formatting, and saves the cleaned data for downstream tasks. The script also provides instructions 
for obtaining the necessary Census API key if the raw data is not present.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from paths import DATA_RAW, DATA_PROCESSED, validate_and_alert
from utils import get_file_hash

census_instructions = """
1. Open your R environment.
2. Run the script: 'pipeline/Rsrc/04_get_census_acs.R'.
3. Ensure the output is saved to 'data/raw/census_acs_county_2023.csv'.
4. Re-run this Python pipeline.

NOTE: To ingest census data you will need an API key from the Census Bureau.
Get one here: https://api.census.gov/data/key_signup.html
In your R console, run the following command:
census_api_key('YOUR_KEY_HERE', install = TRUE, overwrite = TRUE) 
"""

def census_clean():
    # Validate existence using the shared utility
    data_path = DATA_RAW / "census_acs_county_2023.csv"
    validate_and_alert(data_path, "Census ACS", census_instructions)

    # Fingerprint Check
    print(f"🔍 Data Fingerprint: {get_file_hash(data_path)}")
    
    # Proceed with cleaning (No more if/else for existence!)
    print(f"🚀 Processing {data_path}...")
    df = pd.read_csv(data_path, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)
    # Ensure directories exist
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Drop rows with missing income
    df = df.dropna(subset=["median_income"])

    # FIPS Cleaning Step for master Join 
    df = df.rename(columns={'fips': 'fipscode'})

    # --- Ensure padding is applied to the newly renamed column ---
    df['fipscode'] = df['fipscode'].str.zfill(5)

    # Save locally
    output_path = DATA_PROCESSED / "census_acs_county_2023.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Census ACS data saved to {output_path}")

    print("------------------------------------------------------------------")
    print("🚀 04_process_census_acs.py completed successfully. Moving to next task...")
    print("------------------------------------------------------------------")

    return df

if __name__ == "__main__":
    census_clean()
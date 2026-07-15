""" 
This script merges multiple processed datasets into a single master dataset for analysis.
It ensures that the 'fipscode' column is consistently formatted across all datasets, 
handles missing values, and saves the final merged dataset for downstream tasks.
"""

import pandas as pd
from functools import reduce
from paths import DATA_PROCESSED

def create_master_dataset():
    print("🚀 Starting Master Merge...")
    
    # Define paths - Ensure these match your actual filenames
    files = {
        'cdc': DATA_PROCESSED / "processed_cdc_data.csv",
        'usda': DATA_PROCESSED / "processed_usda_county_data.csv",
        'census': DATA_PROCESSED / "census_acs_county_2023.csv",
    }
    
    dfs = []
    for name, path in files.items():
        if path.exists():
            print(f"Loading {name} from {path.name}...")
            # Enforce fipscode as a 5-digit string immediately
            df = pd.read_csv(path, dtype={'fipscode': str})
            df['fipscode'] = df['fipscode'].str.zfill(5)
            dfs.append(df)
        else:
            print(f"⚠️ Warning: {path} not found! Skipping...")

    # Merge all files on 'fipscode'
    # 'outer' join is safe for keeping all counties
    master_df = reduce(lambda left, right: pd.merge(left, right, on='fipscode', how='outer'), dfs)
    
    # Cleanup: Remove duplicate columns (e.g., if 'State' or 'County' appears in multiple files)
    # This keeps the first instance and drops subsequent ones
    master_df = master_df.loc[:, ~master_df.columns.duplicated()]
    
    # Final Formatting
    master_df = master_df.sort_values('fipscode').reset_index(drop=True)

    # Final Integrity Check
    # Check if we have any total duplicates on fipscode
    if master_df['fipscode'].duplicated().any():
        print("⚠️ WARNING: Duplicate FIPS codes found in merged dataset! Please check your source files.")
    
    # Optional: Fill primary key gaps if necessary (e.g., if a file didn't merge)
    master_df = master_df.dropna(subset=['fipscode'])
    
    print(f"✅ Master dataset has {master_df['fipscode'].nunique()} unique counties.")
    
    # Save the master dataset
    output_path = DATA_PROCESSED / "master_dataset_all_variables.csv"
    master_df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Master dataset saved to {output_path}")
    print(f"Final dataset dimensions: {master_df.shape[0]} rows, {master_df.shape[1]} columns")

if __name__ == "__main__":
    create_master_dataset()
"""
This script processes the County Health Rankings & Roadmaps (CHR&R) data, focusing on:
1. Loading the raw CSV data.
2. Cleaning and transforming the data, including handling missing values and reliability flags.
3. Extracting metadata and feature sets.
The final cleaned datasets are saved for downstream analysis.
"""

import pandas as pd
import numpy as np 
import seaborn as sns
from paths import DATA_RAW, DATA_PROCESSED, VALIDATION_DIR, FIGURES_DIR, validate_and_alert
from utils import get_file_hash, get_missingness_summary

chrr_instructions = """
1. Go to: https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation
2. Under the '2025 Annual Data Release' section, download the '2025 CHR CSV Analytic Data'.
3. Move the file into: 'data/raw/'.
4. Rename it to: 'analytic_data2025.csv'.
5. Re-run this Python pipeline.
"""

def load_data(path):
    print(f"🚀 Loading CHR&R data from: {path}")
    df = pd.read_csv(path, header=1, dtype={'fipscode': str}, low_memory=False)
    df['fipscode'] = df['fipscode'].str.zfill(5)
    return df

def process_data_reliability(df):
    """
    Consolidated, case-insensitive function to process all '_flag' columns 
    and clean their corresponding 'RawValue' columns.
    """
    flag_cols = [c for c in df.columns if c.lower().endswith('_flag')]
    
    print(f"🚀 Processing reliability for {len(flag_cols)} flag columns...")
    
    for col in flag_cols:
        df[f'{col}_suppressed'] = (df[col] == 2).astype(int)
        df[f'{col}_unreliable'] = (df[col] == 1).astype(int)
        
        raw_val_col = col.replace('_flag', '_rawvalue')
        
        if raw_val_col in df.columns:
            df.loc[df[col] == 2, raw_val_col] = -1
            df[raw_val_col] = pd.to_numeric(df[raw_val_col], errors='coerce')
            
    print(f"✅ Reliability processing complete.")
    return df

def drop_and_fix_missingness(df):
    """
    Cleans the dataframe by auditing missingness, 
    creating indicators for imputed values, 
    and dropping high-missingness columns.
    """
    missing_pct = df.isnull().mean() * 100
    
    print("--- Audit Report ---")
    get_missingness_summary(df) 
    
    cols_to_drop = missing_pct[missing_pct >= 60].index.tolist()
    cols_to_fix = [c for c in missing_pct.index if 0 < missing_pct[c] < 60]

    df_clean = df.copy()
    new_indicators = {}

    for col in cols_to_fix:
        is_missing = (df_clean[col].isnull() | (df_clean[col] == -1)).astype(int)
        new_indicators[f'{col}_is_missing'] = is_missing
        
        median_val = df_clean[df_clean[col] != -1][col].median()
        df_clean[col] = df_clean[col].replace(-1, np.nan).fillna(median_val)

    df_indicators = pd.DataFrame(new_indicators, index=df_clean.index)
    df_clean = pd.concat([df_clean, df_indicators], axis=1)
    
    df_final = df_clean.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"✅ Cleaning complete. Dropped {len(cols_to_drop)} columns.")
    return df_final


def metadata_extraction(df):
    """
    Extracts metadata columns based on specific patterns.
    Returns: df_metadata, df_features
    """
    metadata_terms = ['cilow', 'cihigh', 'numerator', 'denominator']
    metadata_cols = [c for c in df.columns if any(s in c for s in metadata_terms)]
    flag_indicator_cols = [c for c in df.columns if '_suppressed' in c or '_unreliable' in c]

    df_metadata = df[['fipscode'] + metadata_cols].copy()
    feature_cols = [c for c in df.columns if c not in metadata_cols and c not in flag_indicator_cols and c != 'fipscode']
    df_features = df[['fipscode'] + feature_cols + flag_indicator_cols].copy()
    
    return df_metadata, df_features

def clean_data(df):
    # Reliability processing
    df_cleaned = process_data_reliability(df)

    # Extract Metadata and Features 
    df_metadata, df_features = metadata_extraction(df_cleaned)

    # Preserve the signal before overwriting
    missing_mask = (df_features['v001_rawvalue'].isnull()) | (df_features['v001_rawvalue'] == -1)
    df_features['v001_is_missing'] = missing_mask.astype(int)
    df_features['v001_rawvalue'] = df_features['v001_rawvalue'].replace(-1, np.nan).fillna(df_features['v001_rawvalue'].median())

    # Drop and fix missingness
    df_final = drop_and_fix_missingness(df_features)

    # Handle flag columns 
    race_flag_cols = ['v001_race_aian_flag', 'v001_race_asian_flag', 
                      'v001_race_black_flag', 'v001_race_hispanic_flag', 'v001_race_nhopi_flag']

    for col in race_flag_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].replace({1: 0, 2: 1})
   
    return df_final, df_metadata, df_features

def main():
    raw_path = DATA_RAW / "analytic_data2025.csv"
    
    validate_and_alert(raw_path, "CHR&R Data", chrr_instructions)
    print(f"🔍 Data Fingerprint: {get_file_hash(raw_path)}")
    
    df = load_data(raw_path)
    df_final, df_metadata, df_features = clean_data(df)

    cols = ['fipscode'] + [c for c in df_final.columns if c != 'fipscode']
    df_final = df_final[cols]
    df_final = df_final.sort_values('fipscode')
    
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    df_metadata.to_csv(VALIDATION_DIR / "chrr_metadata_backup.csv", index=False)
    df_features.to_csv(VALIDATION_DIR / "chrr_features_cleaned.csv", index=False)
    df_final.to_csv(VALIDATION_DIR / "chrr_final_cleaned.csv", index=False)
    print("✅ Files saved successfully.")

    print("------------------------------------------------------------------")
    print("🚀 03_process_chrr.py completed successfully. Moving to next task...")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    main()
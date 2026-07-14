import pandas as pd
import numpy as np 
import seaborn as sns
from paths import DATA_RAW, DATA_PROCESSED, VALIDATION_DIR, FIGURES_DIR
from utils import get_file_hash, get_missingness_summary

def load_data():
   # Ensure directories exist
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load
    raw_data_path = DATA_RAW / "analytic_data2025.csv"
    if raw_data_path.exists():
        print(f"File fingerprint: {get_file_hash(raw_data_path)}")
    df = pd.read_csv(raw_data_path, header=1, dtype={'fipscode': str}, low_memory=False)
    # Pad with a leading zero if it's only 4 digits (e.g., '1001' -> '01001')
    df['fipscode'] = df['fipscode'].str.zfill(5)

    # Verify integrity before cleaning
    current_hash = get_file_hash(raw_data_path)
    print(f"CDC Data Fingerprint: {current_hash}")
    return df

def process_data_reliability(df):
    """
    Consolidated, case-insensitive function to process all '_flag' columns 
    and clean their corresponding 'RawValue' columns.
    """
    # 1. Detect columns ending in '_flag' (case-insensitive to be safe)
    flag_cols = [c for c in df.columns if c.lower().endswith('_flag')]
    
    print(f"🚀 Processing reliability for {len(flag_cols)} flag columns...")
    
    for col in flag_cols:
        # Create binary indicator for suppression (Flag == 2)
        df[f'{col}_suppressed'] = (df[col] == 2).astype(int)
        
        # Create binary indicator for unreliable (Flag == 1)
        df[f'{col}_unreliable'] = (df[col] == 1).astype(int)
        
        # 2. Link to RawValue and apply placeholder
        # We replace '_flag' with '_rawvalue' to find the corresponding column
        raw_val_col = col.replace('_flag', '_rawvalue')
        
        if raw_val_col in df.columns:
            # Mask suppressed values (Flag == 2) with -1
            df.loc[df[col] == 2, raw_val_col] = -1
            
            # Ensure RawValue is numeric
            df[raw_val_col] = pd.to_numeric(df[raw_val_col], errors='coerce')
            
    print(f"✅ Reliability processing complete.")
    return df

def drop_and_fix_missingness(df):
    """
    Cleans the dataframe by auditing missingness, 
    creating indicators for imputed values, 
    and dropping high-missingness columns.
    """
    # Calculate missingness 
    missing_pct = df.isnull().mean() * 100
    
    # Audit (Using your utility)
    print("--- Audit Report ---")
    get_missingness_summary(df) 
    
    # Identify tiers
    cols_to_drop = missing_pct[missing_pct >= 60].index.tolist()
    cols_to_fix = [c for c in missing_pct.index if 0 < missing_pct[c] < 60]

    # Process Fixes
    # We work on a copy to avoid SettingWithCopy warnings
    df_clean = df.copy()
    new_indicators = {}

    for col in cols_to_fix:
        # Create missingness indicator (capturing NaNs and your -1 placeholders)
        is_missing = (df_clean[col].isnull() | (df_clean[col] == -1)).astype(int)
        new_indicators[f'{col}_is_missing'] = is_missing
        
        # Impute: replace -1 with NaN, then fill with median
        median_val = df_clean[df_clean[col] != -1][col].median()
        df_clean[col] = df_clean[col].replace(-1, np.nan).fillna(median_val)

    # Finalize
    # Add all new indicators at once
    df_indicators = pd.DataFrame(new_indicators, index=df_clean.index)
    df_clean = pd.concat([df_clean, df_indicators], axis=1)
    
    # Drop the high-missingness columns
    df_final = df_clean.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"✅ Cleaning complete. Dropped {len(cols_to_drop)} columns.")
    return df_final


def metadata_extraction(df):
    """
    Extracts metadata columns based on specific patterns.
    Returns: df_metadata, df_features
    """
    # 1. Define lists (using set for faster lookup)
    metadata_terms = ['cilow', 'cihigh', 'numerator', 'denominator']
    metadata_cols = [c for c in df.columns if any(s in c for s in metadata_terms)]
    flag_indicator_cols = [c for c in df.columns if '_suppressed' in c or '_unreliable' in c]

    # 2. Extract Metadata
    df_metadata = df[['fipscode'] + metadata_cols].copy()

    # 3. Extract Features
    # Note: We filter out metadata and keep indicator flags with features
    feature_cols = [c for c in df.columns if c not in metadata_cols and c not in flag_indicator_cols and c != 'fipscode']
    df_features = df[['fipscode'] + feature_cols + flag_indicator_cols].copy()
    
    return df_metadata, df_features

def clean_data(df):

   # Reliability processing
   df_cleaned = process_data_reliability(df)

    # Extract Metadata and Features 
   df_metadata, df_features = metadata_extraction(df_cleaned)

   # Preserve the signal before overwriting
   # The bulletproof way to catch both NaNs and placeholders
   missing_mask = (df_features['v001_rawvalue'].isnull()) | (df_features['v001_rawvalue'] == -1)
   # apply the fix
   # Impute the missing values with the median
   df_features['v001_is_missing'] = missing_mask.astype(int)
   df_features['v001_rawvalue'] = df_features['v001_rawvalue'].replace(-1, np.nan).fillna(df_features['v001_rawvalue'].median())

   # Drop and fix missingness
   df_final = drop_and_fix_missingness(df_features)

   # Handle flag columns 
   # ONLY target the specific race flags that have 1s and 2s
   race_flag_cols = ['v001_race_aian_flag', 'v001_race_asian_flag', 
   'v001_race_black_flag', 'v001_race_hispanic_flag', 'v001_race_nhopi_flag']

   # Map only 1->0 and 2->1 for these specific columns
   for col in race_flag_cols:
      df_final[col] = df_final[col].replace({1: 0, 2: 1})
   
   return df_final, df_metadata, df_features

def main():
    df = load_data()
    # Unpack the three dataframes
    df_metadata, df_features, df_final = clean_data(df)
    
    processed_path = VALIDATION_DIR # Use path from paths.py
    processed_path.mkdir(parents=True, exist_ok=True)
    
    df_metadata.to_csv(processed_path / "chrr_metadata_backup.csv", index=False)
    df_features.to_csv(processed_path / "chrr_features_cleaned.csv", index=False)
    df_final.to_csv(processed_path / "chrr_final_cleaned.csv", index=False)
    print("✅ Files saved successfully.")

if __name__ == "__main__":
    main()
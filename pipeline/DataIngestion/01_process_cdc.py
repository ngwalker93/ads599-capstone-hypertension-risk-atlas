"""
This script processes the CDC PLACES dataset, ensuring that the data is cleaned, transformed, 
and visualized for further analysis. It checks for the existence of the raw data file, processes it, 
and generates visualizations for key health indicators. The cleaned data is then saved for downstream tasks.
"""

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Prevents the "intrinsic size" warning
import matplotlib.pyplot as plt
import numpy as np
from paths import DATA_RAW, DATA_PROCESSED, FIGURES_DIR, validate_and_alert
from utils import get_file_hash

# Define instructions for the CDC dataset
cdc_instructions = """
1. Open your R environment.
2. Run the script: 'pipeline/Rsrc/01_get_cdc_places.R'.
3. Ensure the output is saved to 'data/raw/cdc_places_aa_raw.csv'.
4. Re-run this Python pipeline.
"""

def clean_cdc():
    # Validate existence using the shared utility
    raw_path = DATA_RAW / "cdc_places_aa_raw.csv"
    validate_and_alert(raw_path, "CDC PLACES", cdc_instructions)
    
    # Proceed with cleaning (No more if/else for existence!)
    print(f"🚀 Processing {raw_path}...")
    df = pd.read_csv(raw_path, dtype={'data_value_footnote_symbol': str, 'data_value_footnote': str})
    # Ensure directories exist
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load
    raw_data_path = DATA_RAW / "cdc_places_aa_raw.csv"
    if raw_data_path.exists():
        print(f"File fingerprint: {get_file_hash(raw_data_path)}")
    df = pd.read_csv(raw_data_path, dtype={'data_value_footnote_symbol': str, 'data_value_footnote': str})

    # Verify integrity before cleaning
    current_hash = get_file_hash(raw_data_path)
    print(f"CDC Data Fingerprint: {current_hash}")

    # Clean
    df = df.drop(columns=['data_value_footnote', 'data_value_footnote_symbol', 'low_confidence_limit', 'high_confidence_limit'], errors='ignore')
    df = df.dropna(subset=['data_value'])
    df = df.drop(columns=['categoryid', 'datavaluetypeid', 'short_question_text', 'geolocation', 'datasource'], errors='ignore')

    # Select Measures
    selected_measures = ['BPHIGH', 'DIABETES', 'OBESITY', 'CSMOKING', 'LPA', 'SLEEP', 'LACKTRPT', 
                         'FOODINSECU', 'HOUSINSECU', 'FOODSTAMP', 'ACCESS2', 'GHLTH', 'CASTHMA', 
                         'COPD', 'LONELINESS', 'EMOTIONSPT', 'MOBILITY', 'COGNITION', 'SELFCARE', 
                         'TEETHLOST', 'VISION']

    # Filter your dataframe
    df_filtered = df[df['measureid'].isin(selected_measures)]

    # Pivot the data to create columns for each measure
    df = df_filtered.pivot_table(
        index=['locationid', 'locationname', 'stateabbr'], 
        columns='measureid', 
        values='data_value'
    ).reset_index()

    # After the pivot_table, ensure we have all selected measures
    missing = [m for m in selected_measures if m not in df.columns]
    if missing:
        # Raise an error to stop the pipeline before you train a model on bad data
        raise ValueError(f"CRITICAL: The following measures were not found after pivot: {missing}")

    # Apply padding immediately so it is ready for all downstream tasks
    df['locationid'] = df['locationid'].astype(str).str.zfill(5)

    # Tell Python these columns are now floats
    for col in selected_measures:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Visualization: Distribution of BPHIGH
    missing_cols = [col for col in selected_measures if col not in df.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing from the dataframe: {missing_cols}")
    else:
        # Now run your plotting code here
        plt.figure(figsize=(10, 6))
        sns.histplot(df['BPHIGH'], kde=True, color='red')
        plt.title('Distribution of BPHIGH across U.S. Counties')
        plt.savefig(FIGURES_DIR / "bphigh_distribution.png")
        plt.close() # Close plot to free memory

    # Visualization: Correlation Heatmap
    corr = df[selected_measures].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Feature Correlation Heatmap: Health & SDoH Indicators')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlation_heatmap.png")
    plt.close()

    # FIPS Cleaning Step for master Join 
    df = df.rename(columns={'locationid': 'fipscode'})

    # Save Processed Data
    df.to_csv(DATA_PROCESSED / "processed_cdc_data.csv", index=False)
    print(f"✅ Cleaned data and figures saved to {DATA_PROCESSED} and {DATA_PROCESSED}")

    print("------------------------------------------------------------------")
    print("🚀 01_process_cdc.py completed successfully. Moving to next task...")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    clean_cdc()
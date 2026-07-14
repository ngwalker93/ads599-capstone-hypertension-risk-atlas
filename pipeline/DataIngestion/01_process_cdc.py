import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Prevents the "intrinsic size" warning
import matplotlib.pyplot as plt
import numpy as np
import requests
from paths import DATA_RAW, DATA_PROCESSED, FIGURES_DIR
from utils import get_file_hash

def get_cdc_data():
    # Define paths and URLs
    file_path = DATA_RAW / "cdc_places_aa_raw.csv"
    base_url = "https://data.cdc.gov/resource/swc5-untb.csv"
    
    # URL encoded query for Age-adjusted prevalence
    # Socrata uses SoQL query language; adding it as a parameter
    params = {
        "$where": "data_value_type='Age-adjusted prevalence'"
    }
    
    # Defensive Check
    if file_path.exists():
        print("🚀 Local CDC PLACES cache found! Loading from disk...")
        return pd.read_csv(file_path)
    
    else:
        print("🌐 Connecting to CDC PLACES API... Fetching filtered data...")
        try:
            # Fetch data from API
            response = requests.get(base_url, params=params)
            response.raise_for_status() # Raise error for bad network calls
            
            # Save to raw folder
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Filtered CDC PLACES dataset saved to {file_path}")
            return pd.read_csv(file_path)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to API: {e}")
            raise

def clean_cdc():

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

    # Save Processed Data
    df.to_csv(DATA_PROCESSED / "processed_cdc_data.csv", index=False)
    print(f"✅ Cleaned data and figures saved to {DATA_PROCESSED} and {DATA_PROCESSED}")

    print("------------------------------------------------------------------")
    print("🚀 01_process_cdc.py completed successfully. Moving to next task...")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    clean_cdc()
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Prevents the "intrinsic size" warning
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_project_root():
    current_path = Path.cwd()
    # Keep going up until we find our project folder
    while current_path.name != "ADS599_Capstone_Hypertension_Risk_Atlas":
        if current_path == current_path.parent: # Reached the system root
            raise FileNotFoundError("Could not find project root folder!")
        current_path = current_path.parent
    return current_path

# Now define your paths using this function
base_path = get_project_root()
raw_path = base_path / "data" / "raw" / "cdc_places_aa_raw.csv"
processed_dir = base_path / "data" / "processed"
figures_dir = base_path / "data" / "figures"  

# Debugging: Print to verify the location
print(f"DEBUG: Looking for raw data at: {raw_path}")
print(f"DEBUG: Saving processed data to: {processed_dir}")

def clean_cdc():

    # Ensure directories exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    df = pd.read_csv(raw_path, dtype={'data_value_footnote_symbol': str, 'data_value_footnote': str})

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
        plt.savefig(figures_dir / "bphigh_distribution.png")
        plt.close() # Close plot to free memory

    # Visualization: Correlation Heatmap
    corr = df[selected_measures].corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title('Feature Correlation Heatmap: Health & SDoH Indicators')
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_correlation_heatmap.png")
    plt.close()

    # Save Processed Data
    df.to_csv(processed_dir / "processed_cdc_data.csv", index=False)
    print(f"✅ Cleaned data and figures saved to {processed_dir} and {figures_dir}")

    print("------------------------------------------------------------------")
    print("🚀 01_process_cdc.py completed successfully. Moving to next task...")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    clean_cdc()
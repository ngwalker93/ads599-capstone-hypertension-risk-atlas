"""
This script performs the first step of feature engineering where counties are 
removed for missing target values, features are synthesized to prevent multicollinearity 
issues, and la_family features are dropped based on the criteria that they are not 
'share' or 'Tract' columns and are not 'number' type.
"""

import pandas as pd 
from paths import DATA_PROCESSED, validate_and_alert

def feature_engineering():

    # Load the processed data
    data_path = DATA_PROCESSED / "master_dataset_all_variables.csv"
    validate_and_alert(data_path, "Master Dataset", "Run Data Ingestion and Cleaning scripts to generate the master dataset.")
    print(f"🚀 Processing {data_path}...")

    # Read FIPS code as a a zero-padded string
    FIPS_COL = "fipscode"
    df = pd.read_csv(data_path, dtype={FIPS_COL: str}, low_memory=False)

    # Dropping rows with missing target values
    # removing 275 counties (8.5%), leaving 2,956 counties for modeling.
    TARGET_COL = "BPHIGH"
    df.dropna(subset=[TARGET_COL], inplace=True)

    # Create the synthesized feature of median_income (from Census ACS Datasest) and MedianFamilyIncome (from USDA Food Access Dataset)
    df['consensus_income'] = (df['median_income'] + df['MedianFamilyIncome']) / 2

    # Assign columns to their source datasets for clarity
    id_cols = ["fipscode", "locationname", "stateabbr", "State", "County"]
    acs_cols = ["poverty_count", "median_income"]
    cdc_cols = ["ACCESS2", "BPHIGH", "CASTHMA", "COGNITION", "COPD", "CSMOKING",
    "DIABETES", "EMOTIONSPT", "FOODINSECU", "FOODSTAMP", "GHLTH","HOUSINSECU", "LACKTRPT", "LONELINESS", "LPA", "MOBILITY",
    "OBESITY", "SELFCARE", "SLEEP", "TEETHLOST", "VISION"]
    usda_cols = [c for c in df.columns if c not in id_cols + acs_cols + cdc_cols]

    # Define the family of features related to "la" and "tract" for VIF calculation
    la_family = [c for c in df.columns if c.lower().startswith(("la", "tract"))]

    # Define "keep" criteria
    # Keep if it contains 'share' OR if it is a 'Tract' column (since these are counts 
    # but are distinct from the 'la' counts), AND explicitly NOT a 'number' type.
    # Note: Since column names contain 'number' as a suffix or indicator, 
    # we exclude them.
    la_family_refined = [
        col for col in la_family 
        if ('share' in col or 'Tract' in col) 
        and 'number' not in col.lower()
    ]

    # Get list of columns to drop based on the refined criteria
    la_cols_to_drop = [col for col in la_family if col not in la_family_refined]

    # Perform the drop on the main dataframe
    df = df.drop(columns=la_cols_to_drop)

    # Save the engineered features to a new CSV file
    df.to_csv(DATA_PROCESSED / "eng_feat.csv", index=False)
    print(f"✅ Engineered Features saved to {DATA_PROCESSED}")

    print("------------------------------------------------------------------")
    print("🚀 01_feature_engineering.py completed successfully. Moving to train-test split...")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    feature_engineering()
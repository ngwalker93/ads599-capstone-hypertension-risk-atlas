"""
This script performs a train-test split on the engineered features dataset. 

Train-test split is performed at this step prior to PCA analysis 
to ensure that the PCA is fit only on the training data, preventing data leakage and 
ensuring that the model's performance is evaluated on unseen data.
"""

from sklearn.model_selection import train_test_split
from paths import DATA_PROCESSED, DATA_FINAL, validate_and_alert
import pandas as pd

def train_test_split_engineered_features():
    # Load the engineered features dataset
    data_path = DATA_PROCESSED / "eng_feat.csv"
    validate_and_alert(data_path, "Engineered Features Dataset", "Run the Feature Engineering script to generate the engineered features dataset.")
    print(f"🚀 Processing {data_path}...")

    # Read the dataset
    df = pd.read_csv(data_path, low_memory=False)

    # Define the target column
    TARGET_COL = "BPHIGH"
    
    # Train, test, and validation split 
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=[TARGET_COL]), df[TARGET_COL], test_size=0.2, random_state=42)

    # --- SAFETY ASSERTION ---
    # This will crash the script instantly if the target ever sneaks back into X
    assert TARGET_COL not in x_train.columns, f"ERROR: Target column '{TARGET_COL}' is present in x_train!"

    # Save the split datasets (wrapping in pd.DataFrame/Series guarantees .to_csv() works and satisfies Pyright)
    pd.DataFrame(x_train).to_csv(DATA_FINAL / "x_train.csv", index=False)
    pd.DataFrame(x_test).to_csv(DATA_FINAL / "x_test.csv", index=False)
    pd.Series(y_train).to_csv(DATA_FINAL / "y_train.csv", index=False)
    pd.Series(y_test).to_csv(DATA_FINAL / "y_test.csv", index=False)
    
    print(f"✅ Train-test splits saved successfully to {DATA_FINAL}")

    print("------------------------------------------------------------------")
    print("🚀 02_train_test_split.py completed successfully. Moving to PCA Feature Engineering...")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    train_test_split_engineered_features()
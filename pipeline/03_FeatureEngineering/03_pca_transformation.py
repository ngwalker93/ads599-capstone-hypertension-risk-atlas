"""
This script performs Principal Component Analysis (PCA) on the refined 
food access (la_family) features. It fits the PCA exclusively on the 
training data to prevent data leakage, transforms both train and test sets, 
and replaces the raw columns with the new PCA components.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from paths import DATA_FINAL, DATA_PROCESSED, validate_and_alert

def apply_pca():
    # 1. Load the train and test split datasets from DATA_PROCESSED
    x_train_path = DATA_PROCESSED / "x_train.csv"
    x_test_path = DATA_PROCESSED / "x_test.csv"
    y_train_path = DATA_PROCESSED / "y_train.csv"
    y_test_path = DATA_PROCESSED / "y_test.csv"
    
    validate_and_alert(x_train_path, "X Train Dataset", "Run the train-test split script first.")
    validate_and_alert(x_test_path, "X Test Dataset", "Run the train-test split script first.")
    validate_and_alert(y_train_path, "Y Train Dataset", "Run the train-test split script first.")
    validate_and_alert(y_test_path, "Y Test Dataset", "Run the train-test split script first.")
    
    print(f"🚀 Loading split datasets from {DATA_PROCESSED}...")
    x_train = pd.read_csv(x_train_path, low_memory=False, dtype={"fipscode": str})
    x_test = pd.read_csv(x_test_path, low_memory=False, dtype={"fipscode": str})
    y_train = pd.read_csv(y_train_path, low_memory=False).squeeze("columns")
    y_test = pd.read_csv(y_test_path, low_memory=False).squeeze("columns")

    # Re-identify la_family_refined columns matching feature engineering rules
    la_family = [c for c in x_train.columns if c.lower().startswith(("la", "tract"))]
    la_family_refined = [
        col for col in la_family 
        if ('share' in col or 'Tract' in col) 
        and 'number' not in col.lower()
    ]

    print(f"Identified {len(la_family_refined)} refined la_family features for PCA.")

    # Handle potential NaNs safely for PCA subset features
    train_subset = x_train[la_family_refined].dropna()
    valid_train_indices = train_subset.index
    current_y_train = y_train.loc[valid_train_indices]

    test_subset = x_test[la_family_refined].dropna()
    valid_test_indices = test_subset.index
    current_y_test = y_test.loc[valid_test_indices]

    # 2. Initialize and Fit PCA on Training Data Only (prevent data leakage)
    n_components = 2
    la_pca = PCA(n_components=n_components, random_state=42)
    
    x_train_pca = la_pca.fit_transform(train_subset)
    x_test_pca = la_pca.transform(test_subset)
    
    # Create readable column names for the PCA components
    pca_columns = [f'PCA_Component_{i+1}' for i in range(n_components)]

    # Convert results back into DataFrames with original valid indices
    x_train_pca_df = pd.DataFrame(x_train_pca, columns=pca_columns, index=valid_train_indices)
    x_test_pca_df = pd.DataFrame(x_test_pca, columns=pca_columns, index=valid_test_indices)

    # 3. Reconstruct final datasets by dropping raw refined columns and adding PCA components
    x_train_final = x_train.loc[valid_train_indices].drop(columns=la_family_refined)
    x_test_final = x_test.loc[valid_test_indices].drop(columns=la_family_refined)

    x_train_final = pd.concat([x_train_final, x_train_pca_df], axis=1)
    x_test_final = pd.concat([x_test_final, x_test_pca_df], axis=1)

    # Update y labels to match filtered indices
    y_train_final = current_y_train
    y_test_final = current_y_test

    # 4. Save the final modeling-ready datasets back to DATA_FINAL
    DATA_FINAL.mkdir(parents=True, exist_ok=True)
    
    x_train_final.to_csv(DATA_FINAL / "X_train.csv", index=False)
    x_test_final.to_csv(DATA_FINAL / "X_test.csv", index=False)
    y_train_final.to_csv(DATA_FINAL / "y_train.csv", index=False)
    y_test_final.to_csv(DATA_FINAL / "y_test.csv", index=False)

    print(f"✅ PCA-transformed datasets saved successfully to {DATA_FINAL}")
    print(f"Final training set shape: {x_train_final.shape}")
    print(f"Final testing set shape: {x_test_final.shape}")

    # Print Loadings for reporting verification
    loadings = pd.DataFrame(la_pca.components_.T, columns=pca_columns, index=la_family_refined)
    print("\nTop features contributing to PCA Component 1:")
    print(loadings.sort_values(by='PCA_Component_1', ascending=False).head(5))

    print("------------------------------------------------------------------")
    print("🚀 03_pca_transformation.py completed successfully.")
    print("------------------------------------------------------------------")

if __name__ == "__main__":
    apply_pca()
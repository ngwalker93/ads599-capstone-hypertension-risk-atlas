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
from sklearn.impute import SimpleImputer
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

    # Initialize the imputer (median is great for skewed census/health data)
    imputer = SimpleImputer(strategy='median')

    # Impute the la_family columns (Fit on train ONLY, transform both)
    train_la_imputed = imputer.fit_transform(x_train[la_family_refined])
    test_la_imputed = imputer.transform(x_test[la_family_refined])

    # Convert back to DataFrames to keep column names and indices clean
    train_la_df = pd.DataFrame(train_la_imputed, columns=la_family_refined, index=x_train.index)
    test_la_df = pd.DataFrame(test_la_imputed, columns=la_family_refined, index=x_test.index)

    # Initialize and Fit PCA on Training Data Only (prevent data leakage)
    n_components = 2
    la_pca = PCA(n_components=n_components, random_state=42)
    
    x_train_pca = la_pca.fit_transform(train_la_df)
    x_test_pca = la_pca.transform(test_la_df)
    
    # Create readable column names for the PCA components
    pca_columns = [f'PCA_Component_{i+1}' for i in range(la_pca.n_components_)]

    # Convert results back into DataFrames with original valid indices
    x_train_pca_df = pd.DataFrame(x_train_pca, columns=pca_columns, index=train_la_df.index)
    x_test_pca_df = pd.DataFrame(x_test_pca, columns=pca_columns, index=test_la_df.index)

    # Ensure the indices match up with your targets
    assert x_train_pca_df.index.equals(y_train.index)
    assert x_test_pca_df.index.equals(y_test.index)

    # Reconstruct final datasets by dropping raw refined columns and adding PCA components
    x_train_final = x_train.drop(columns=la_family_refined)
    x_test_final = x_test.drop(columns=la_family_refined)

    x_train_final = pd.concat([x_train_final, x_train_pca_df], axis=1)
    x_test_final = pd.concat([x_test_final, x_test_pca_df], axis=1)

    # isolate numeric columns for any further analysis or checks
    numeric_cols = x_train_final.select_dtypes(include=[np.number]).columns

    # Use an imputer to keep full row count safe
    final_imputer = SimpleImputer(strategy='median')

    # Apply imputer only to the numeric columns, keeping original DataFrames intact for non-numeric columns
    x_train_final[numeric_cols] = final_imputer.fit_transform(x_train_final[numeric_cols])
    x_test_final[numeric_cols] = final_imputer.transform(x_test_final[numeric_cols])

    # Update Y-labels to match the cleaned X-frames
    y_train_final = y_train.loc[x_train_final.index]
    y_test_final = y_test.loc[x_test_final.index]

    # Save the final modeling-ready datasets back to DATA_FINAL
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
This folder integrated data scaling, feature reduction, and modeling steps into a unified structure, moving away from fragmented exploratory scripts. 

Key Files: 

- 01_feature_engineering.py – Implements feature scaling and reduction techniques, preparing the dataset for modeling.
- 02_train_test_split.py – Splits the master dataset into training and testing sets, ensuring reproducibility.
- 03_pca_transformation.py – Applies PCA for dimensionality reduction to the la_family columns to enhance model performance.

How to run: 
1. Clone the repository to your local machine.
2. Navigate to the pipeline/03_FeatureEngineering directory.
3. Run the cleaning scripts:
   - `python 04_run_feature_engineering.py` to execute the entire ingestion and cleaning process.
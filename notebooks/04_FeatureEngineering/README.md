# Feature Engineering & Baseline Modeling Documentation

## Overview

This folder contains the transformations used to create the modeling dataset. The workflow is divided into two phases: feature refinement (Creation, Selection, Reduction) and baseline performance assessment.

## Pipeline Summary

1. `01_feature_engineering.ipynb`:

    Target Variable Integrity: Excluded 275 counties (8.5%) with missing target values. All subsequent feature engineering, LASSO selection, and baseline modeling are performed on the finalized subset of 2,956 counties.

    Data Block Analysis: For the USDA la* and Tract* feature families, "number" columns were dropped to reduce extreme collinearity. The remaining "share" columns were transformed using PCA to derive two latent "Low-Access Indices," effectively capturing demographic variance while ensuring orthogonality.

    Train/Test Split: Implemented stratified sampling based on the target variable to ensure balanced representation in both sets, executed prior to the PCA transformation to prevent data leakage.

2. `02_baseline_modeling.ipynb`:
    Feature Selection: Implemented LASSO ($L1$) regularization to prune non-contributory variables and mitigate multicollinearity. The optimal regularization strength ($\alpha=0.0055$) was determined via cross-validation.
    
    Methodological Rigor: To ensure the integrity of the held-out test set and prevent data leakage, all formal model evaluation is deferred to the final integrated pipeline. All feature selection processes were restricted to the training split to maintain experimental validity.

3. `03_dimensionality_reduction.ipynb`:
        Dimensionality Reduction: Evaluated Principal Component Analysis (PCA) within a cross-validated pipeline to identify an optimal latent feature space. Sensitivity analysis established that 15 components provide the best balance of predictive accuracy ($R^2 \approx 0.87$) and model parsimony.

4. `04_baseline_modeling.ipynb`:
    perform linear regression modeling using the refined feature set. The notebook includes model training, evaluation, and interpretation of results.

Key Artifacts
X_train.csv: Contains the training feature set.
X_test.csv: Contains the test feature set.
y_train.csv: Contains the training target variable.
y_test.csv: Contains the test target variable.
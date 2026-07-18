# Feature Engineering & Baseline Modeling Documentation

## Overview

This folder contains the transformations used to create the modeling dataset. The workflow is divided into two phases: feature refinement (Creation, Selection, Reduction) and baseline performance assessment.

## Pipeline Summary

1. `01_feature_engineering.ipynb`:

    Target Variable Integrity: Excluded 275 counties (8.5%) with missing target values. All subsequent feature engineering, LASSO selection, and baseline modeling are performed on the finalized subset of 2,956 counties.

    Data Block Analysis: For the USDA la* and Tract* feature families, "number" columns were dropped to reduce extreme collinearity. The remaining "share" columns were transformed using PCA to derive two latent "Low-Access Indices," effectively capturing demographic variance while ensuring orthogonality.

    Train/Test Split: Implemented stratified sampling based on the target variable to ensure balanced representation in both sets, executed prior to the PCA transformation to prevent data leakage.




    

Selection: Used LASSO ($L1$ penalty) to drop irrelevant features and reduce multicollinearity.

Reduction: Applied PCA to aggregate correlated socioeconomic variables into latent "driver" components.

`02_baseline_modeling.ipynb`:

    Establishes a performance floor using [insert your baseline model, e.g., Logistic Regression].
    
    Metrics: Accuracy, AUC-ROC, and F1-Score provided as a benchmark for your final analysis.

Key Artifactsmodeling_ready_data.csv: The final dataset passed for model training.feature_importance_report.csv: A summary of the variables retained after LASSO.pca_components_loadings.csv: The coefficients for the PCA components (use these to interpret your new variables!).Note: Data Leakage prevention—Feature selection and PCA transformations were derived based on the training split only.
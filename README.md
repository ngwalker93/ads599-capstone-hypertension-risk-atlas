# ADS599_Project

# Project: Chronic Disease Registry Data Pipeline

### Data Ingestion Pipeline (DataIngestion.qmd)

This file is the automated ETL pipeline for the project. It handles the raw-to-processed transition for all source data.

- Workflow: Ingests raw data $\rightarrow$ Standardizes GEOID keys $\rightarrow$ Aggregates to county-level (1:1 grain) $\rightarrow$ Performs sequential left_join.

- Output: Generates data/processed/master_dataset.csv, the source of truth for the SQL database and downstream ML models

- Execution: Run the full Quarto document to refresh the master dataset.
This folder contains Python cleaning scripts. 

Purpose: ETL (Extract, Transform, Load) scripts that standardize data for the master pipeline.

Key Files: 

- 01_process_cdc.py – Cleans CDC PLACES data.
- 02_process_usda.py – Aggregates Food Atlas data.
- 03_process_census.py – Standardizes ACS metrics.

•  Step A (Python): produce the cleaned CSVs.
•  Step B (Python): A single "Merge script" creating master_dataset.csv.
•  Step C (SQL): Use a script to load finalized master_dataset.csv into a single SQL table.

•  Future Steps (Consumption):
•	For Modeling: Read the master_dataset.csv directly into Python. It is faster and easier for research.
•	For Dashboarding: Connect dashboarding tool to the SQL database.
•  For Validation: Use the validation dataset as an external dataset.

How to run: 
1. Clone the repository to your local machine.
2. Navigate to the pipeline/DataIngestion directory.
3. Run the cleaning scripts:
   - `python 07_run_ingestion.py` to execute the entire ingestion and cleaning process.

Pipeline Outputs: 
 - `master_dataset.csv` – The final cleaned and merged dataset ready for analysis.
 - SQL Relational Database – All datasets loaded into a SQL table for dashboarding and further analysis.
 - Validation Dataset: `chrr_final_cleaned.csv` – A sample dataset to validate the cleaning process.
 - Metadataset: `chrr_metadata_backup.csv` – Contains metadata about the CHR&R dataset used in the pipeline.
 - `chrr_features_cleaned.csv` - Contains cleaned features for the CHR&R dataset, an intermediate step in the cleaning process.
 - bphigh_distrubution.png – Distribution of hypertension prevalence (BPHIGH) across U.S. counties.
 - feature_correlation_heatmap.png - A heatmap visualizing features in the CDC Places Dataset. Correlation heatmap for key health and SDoH indicators.

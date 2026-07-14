"""
STATUS: WORK IN PROGRESS
Project: ADS599 Capstone - Hypertension Risk Atlas
File: 05_load_to_sql.py
Note: This script is currently under development. 
      DO NOT run in production.
"""

import sqlite3
import pandas as pd
from pathlib import Path

def load_processed_to_sql(db_name="capstone_health.db"):
    # 1. Define the directory where your clean CSVs live
    processed_dir = Path.cwd().parents[1] / "data" / "processed"
    conn = sqlite3.connect(db_name)
    
    # 2. Iterate through files and load them
    # This automatically creates a table for every CSV found
    for csv_file in processed_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        table_name = csv_file.stem  # e.g., "chrr_features_cleaned"
        
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"✅ Loaded {csv_file.name} into table '{table_name}'")
        
    conn.close()

# Updated create_master_data.py (The "SQL Orchestrator")
import sqlite3

def create_master_view(db_name="capstone_health.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # This SQL command replaces your manual pd.merge() steps
    sql_query = """
    CREATE VIEW IF NOT EXISTS master_dataset AS
    SELECT 
        c.fipscode,
        c.population,
        ch.raw_value AS chrr_raw,
        u.food_index AS usda_index
    FROM processed_census_acs c
    LEFT JOIN processed_ChRr ch ON c.fipscode = ch.fipscode
    LEFT JOIN processed_usda u ON c.fipscode = u.fipscode
    """
    
    cursor.execute(sql_query)
    conn.commit()
    conn.close()
    print("✅ Master View created in database.")

if __name__ == "__main__":
    load_to_sql()


# write after Validation EDA compleate 

#Instead of merging datasets into one giant CSV, 
# write them into your local SQLite/DuckDB database as individual tables.

# Perform your Master merge using SQL queries.

CREATE VIEW v_Master_Hypertension_Study AS
SELECT 
    c.fipscode,
    c.population,
    cdc.diabetes_rate,
    usda.food_access_index,
    chrr.premature_death_rate
FROM census c
JOIN cdc_data cdc ON c.fipscode = cdc.fipscode
JOIN usda_data usda ON c.fipscode = usda.fipscode
JOIN chrr_data chrr ON c.fipscode = chrr.fipscode;
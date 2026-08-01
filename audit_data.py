"""
Memory Efficient (nrows=0): By reading only the headers (nrows=0), 
inspect dozens of massive CSV files in milliseconds without 
loading gigabytes of data into memory.

Wildcard Globbing (DATA_PROCESSED.glob("*.csv")): 
Automatically sweeps through the processed directory to inspect every 
output table.
"""

import pandas as pd
from paths import DATA_PROCESSED

def audit_columns():
    for file in DATA_PROCESSED.glob("*.csv"):
        df = pd.read_csv(file, nrows=0)  # Only read the header
        print(f"--- {file.name} ---")
        print(list(df.columns))
        print("\n")

if __name__ == "__main__":
    audit_columns()
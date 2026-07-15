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
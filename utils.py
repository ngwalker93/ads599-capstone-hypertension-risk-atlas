from paths import DATA_RAW, DATA_PROCESSED, FIGURES_DIR
import hashlib
import pandas as pd

def get_file_hash(file_path):
    """Generates an MD5 hash of your data file to ensure integrity."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_missingness_summary(df):
    """
    Analyzes a dataframe and returns a summary table of missingness tiers.
    """
    missing_pct = df.isnull().mean() * 100
    
    summary = pd.DataFrame({
        "Missing Threshold": ["0%", "<5%", "5–20%", "20–40%", "40–60%", ">60%"],
        "Count": [
            (missing_pct == 0).sum(),
            ((missing_pct > 0) & (missing_pct < 5)).sum(),
            ((missing_pct >= 5) & (missing_pct < 20)).sum(),
            ((missing_pct >= 20) & (missing_pct < 40)).sum(),
            ((missing_pct >= 40) & (missing_pct < 60)).sum(),
            (missing_pct >= 60).sum()
        ]
    })
    return summary
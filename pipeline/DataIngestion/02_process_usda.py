import pandas as pd
from paths import DATA_RAW, DATA_PROCESSED

def load_data():
    """Handles raw data ingestion using centralized paths."""
    file_path = DATA_RAW / "FoodAccessResearchAtlasData2019.xlsx"
    
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} was not found!")
    
    print(f"🚀 Loading USDA data from: {file_path}")
    return pd.read_excel(file_path, sheet_name='Food Access Research Atlas', dtype={'CensusTract': str})

def handle_missing_values(df):
    """Encapsulates all your missingness/structural cleaning rules."""
    # Drop columns depending on their missingness
    missing_pct = df.isnull().mean() * 100
    # Structural columns follow the la* naming pattern (low-access distance bands)
    structural_cols = [c for c in missing_pct.index if c.lower().startswith('la')]

    genuine_cols = [c for c in missing_pct.index if c not in structural_cols and missing_pct[c] > 0]
    check = pd.DataFrame({'pct_missing': missing_pct})

    def tier_label(p):
        if p == 0:    return '0%'
        elif p < 5:   return '<5%'
        elif p < 20:  return '5-20%'
        elif p < 40:  return '20-40%'
        elif p < 60:  return '40-60%'
        else:         return '>60%'

    check['tier'] = check['pct_missing'].apply(tier_label)
    check['structural'] = check.index.isin(structural_cols)

    # Helpful for debugging, keep it visible!
    print("--- Missingness Summary ---")
    print(pd.crosstab(check['tier'], check['structural']))
    print("---------------------------")

    tier_gt60 = missing_pct[missing_pct >= 60].index.tolist()

    df_clean = df.drop(columns=tier_gt60, errors='ignore')

    tier_40_60 = missing_pct[(missing_pct >= 40) & (missing_pct < 60)].index.tolist()

    # Guardrail: only fill la* columns; anything unexpected stays untouched and visible
    la_40_60 = [c for c in tier_40_60 if c.lower().startswith('la') and c in df_clean.columns]
    df_clean[la_40_60] = df_clean[la_40_60].fillna(0)

    # Columns with 20–40% missing → fill structural blanks with 0
    tier_20_40 = missing_pct[(missing_pct >= 20) & (missing_pct < 40)].index.tolist()

    la_20_40    = [c for c in tier_20_40 if c.lower().startswith('la') and c in df_clean.columns]
    other_20_40 = [c for c in tier_20_40 if c not in la_20_40]

    df_clean[la_20_40] = df_clean[la_20_40].fillna(0)

    # Handle the <5% and 5–20% tiers: impute genuine gaps
    tier_lt20 = missing_pct[(missing_pct > 0) & (missing_pct < 20)].index.tolist()

    la_lt20      = [c for c in tier_lt20 if c.lower().startswith('la') and c in df_clean.columns]
    genuine_lt20 = [c for c in tier_lt20 if c not in la_lt20 and c in df_clean.columns]

    # Structural → 0
    df_clean[la_lt20] = df_clean[la_lt20].fillna(0)

    # Genuine numeric → state median
    for col in genuine_lt20:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean.groupby('State')[col].transform(
                lambda s: s.fillna(s.median())
            )

    return df_clean

def aggregate_to_county(df):
    """Handles the heavy lifting of weighted means and grouping."""
    # Aggregation and profile county level
    df['CountyFIPS'] = df['CensusTract'].str[:5]  

    tracts_per_county = df.groupby('CountyFIPS').size()
    print(tracts_per_county.describe().round(1))

    # Define Aggregation Rules 
    id_cols = ['CensusTract', 'State', 'County', 'CountyFIPS']

    # 0/1 flags: integer columns whose only values are 0 and 1
    flag_cols = [c for c in df.columns if c not in id_cols and df[c].dropna().isin([0, 1]).all()]

    # rates/shares: end in "share" or are known rates
    rate_cols = [c for c in df.columns if c.endswith('share')]
    rate_cols += ['PovertyRate', 'MedianFamilyIncome', 'PCTGQTRS']

    # counts: every remaining numeric column
    count_cols = [c for c in df.columns if c not in id_cols + flag_cols + rate_cols]

    # Aggregate tract county 
    temp = df.copy()
    weight_cols = flag_cols + rate_cols

    # multiply by population
    for col in weight_cols:
        temp[col + '_x_pop'] = temp[col] * temp['Pop2010']

    # one groupby-sum for everything
    sum_cols = count_cols + [c + '_x_pop' for c in weight_cols]
    county = temp.groupby('CountyFIPS')[sum_cols].sum()

    # divide to finish the weighted means
    for col in weight_cols:
        county[col] = county[col + '_x_pop'] / county['Pop2010']
        county = county.drop(columns=col + '_x_pop')

    # bring back state and county names
    county[['State', 'County']] = df.groupby('CountyFIPS')[['State', 'County']].first()
    county = county.reset_index()

    print("County table shape:", county.shape)
    print(county.head())

    key_vars = ['LILATracts_1And10', 'LILATracts_Vehicle', 'PovertyRate',
            'MedianFamilyIncome', 'lapop1share', 'lahunv1share']

    county[key_vars].describe().round(3)
    return county

def process_usda():
    # 1. Setup
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    try:
        # Ingest
        df = load_data()
        
        # Transform
        df_cleaned = handle_missing_values(df)
        county_df = aggregate_to_county(df_cleaned)
    
        # Save
        out_path = DATA_PROCESSED / "processed_usda_county_data.csv"
        county_df.to_csv(out_path, index=False)
        
        print(f"✅ Success! USDA data saved to {out_path}")
        print(f"📊 Final shape: {county_df.shape}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    process_usda()


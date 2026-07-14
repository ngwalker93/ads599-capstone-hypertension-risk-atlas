"""
STATUS: WORK IN PROGRESS
Project: ADS599 Capstone - Hypertension Risk Atlas
File: 04_process_census_acs.py
Note: This script is currently under development. 
      API integration and cleaning logic are in progress.
      DO NOT run in production.
"""

import pandas as pd
from census import Census
from paths import DATA_RAW

# Setup your API Client
# Store this in an .env file, but for now:
API_KEY = "YOUR_KEY_HERE"
c = Census(API_KEY)

def fetch_census_data():
    # 2. Define variables (B19013_001: Median Income, B17001_002: Poverty Count)
    # The Census API uses specific naming conventions for years
    data = c.acs5.state_county(
        fields=('NAME', 'B19013_001', 'B17001_002'), 
        state_fips=Census.ALL, 
        county_fips=Census.ALL,
        year=2023
    )
    
    # 3. Convert to DataFrame
    df = pd.DataFrame(data)
    
    # 4. Create your FIPS code (State + County)
    df['fipscode'] = df['state'] + df['county']
    
    # 5. Rename and Clean
    df = df.rename(columns={
        'B19013_001': 'median_income',
        'B17001_002': 'poverty_count'
    })
    
    # 6. Save locally
    output_path = DATA_RAW / "census_acs_county_2023.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Census ACS data saved to {output_path}")
    return df

# Run it
census_df = fetch_census_data()
print(census_df.head(10))
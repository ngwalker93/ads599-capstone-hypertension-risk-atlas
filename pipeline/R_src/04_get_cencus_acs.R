# ==============================================================================
# PROJECT: ADS599 Capstone - Hypertension Risk Atlas
# SCRIPT: 04_get_census_acs.R
# AUTHOR: Nancy Walker
# DATE: 2026-07-14
# DESCRIPTION: 
#    Fetches raw ACS Census data via tidycensus API and saves to data/raw/
#    for use in the main Python ingestion pipeline.
#
#    To ingest census data you will need an API key from the Census Bureau [@census_acs]
#    Get one here: https://api.census.gov/data/key_signup.html
# 
#    In your R console, run the following command to set your key (replace "YOUR_KEY_HERE" with your actual key):
#    Replace 'YOUR_KEY_HERE' with the key they emailed you
#    census_api_key("YOUR_KEY_HERE", install = TRUE, overwrite = TRUE)
# ==============================================================================

library(RSocrata)   # For CDC PLACES
library(readxl)     # For reading/verifying USDA Excel files
library(tidyverse)  # For data writing utilities (readr)
library(tidycensus) # For Censue Bureau data 
library(knitr)
library(kableExtra)
library(stringr)
library(dplyr)
library(tidyr)

### Ingest U.S. Census Bureau (ACS 5-Year Estimates)

# Define variables we want (e.g., Median Income and Poverty Rate)
census_vars <- c(
  med_income = "B19013_001",
  poverty_rate = "B17001_002"
)

# Fetch data at the County level
message("Fetching Census ACS data...")
census_raw <- get_acs(
  geography = "county",
  variables = census_vars,
  year = 2023,
  survey = "acs5"
)

# Clean and pivot to a "wide" format (One row per FIPS)
census_clean <- census_raw %>%
  select(GEOID, variable, estimate) %>%
  pivot_wider(names_from = variable, values_from = estimate) %>%
  rename(fips = GEOID, 
         median_income = med_income, 
         poverty_count = poverty_rate)


# Render a clean preview table of the Census data
census_clean %>%
  head(10) %>% # Shows only the first 10 rows
  kable(caption = "Census ACS 5-Year Estimates (Sample Data)") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))

# Save the raw, unclearned API response into local storage for reproducibility
write_csv(census_clean, "data/raw/census_acs_county_2023.csv")
message("✅ Census ACS data saved to data/raw/census_acs_county_2023.csv")

# ==============================================================================
# PROJECT: ADS599 Capstone - Hypertension Risk Atlas
# SCRIPT: 01_get_cdc_places.R
# AUTHOR: Nancy Walker
# DATE: 2026-07-14
# DESCRIPTION: 
#    Fetches raw CDC Places data via tidycensus API and saves to data/raw/
#    for use in the main Python ingestion pipeline.
# ==============================================================================

# Import Libraries 
library(RSocrata)   # For CDC PLACES
library(readxl)     # For reading/verifying USDA Excel files
library(tidyverse)  # For data writing utilities (readr)
library(tidycensus) # For Censue Bureau data 
library(knitr)
library(kableExtra)
library(stringr)
library(dplyr)
library(tidyr)

## DIRECTORY MANAGEMENT
# Define Data Folder Stucture
data_folders <- c("data/raw", "data/processed", "data/final")

# Loop through and create only if they don't exist
for (folder in data_folders) {
  if (!dir.exists(folder)) {
    dir.create(folder, recursive = TRUE)
    message(paste0("📁 Created folder: ", folder))
  } else {
    message(paste0("📂 Folder already exists: ", folder))
  }
}

#  DATA DOWNLOAD / INGESTION

## CDC PLACES Data 
# Define file path first
local_cdc_aa_csv <- "data/raw/cdc_places_aa_raw.csv"

# Define your base endpoint (Example ID: swc5-untb for County data)
base_url <- "https://data.cdc.gov/resource/swc5-untb.csv"

# Construct the filter string
# Note: Use backticks for column names and ensure the values match exactly
filter_query <- "?$where=`data_value_type`='Age-adjusted prevalence'"

# Combine into your final endpoint
cdc_endpoint <- paste0(base_url, filter_query)

# Now your defensive check remains the same
if (file.exists(local_cdc_aa_csv)) {
  message(" Local CDC PLACES cache found! Loading data directly from disk...")
  cdc_places_raw <- readr::read_csv(local_cdc_aa_csv, show_col_types = FALSE)
  
} else {
  message("Connecting to CDC PLACES API with filter... Fetching subset...")
  # read.socrata handles the combined URL automatically
  cdc_places_aa_raw <- read.socrata(cdc_endpoint)
  
  readr::write_csv(cdc_places_aa_raw, local_cdc_aa_csv)
  message("Filtered CDC PLACES dataset saved.")
}

# Extract a small subset to verify data parity in the report preview
cdc_test <- cdc_places_aa_raw %>%
  head(5) %>%
  dplyr::select(1:5)

# Render a clean preview table for the PDF report
knitr::kable(cdc_test, caption = "CDC PLACES Sample Data Stream (First 5 Columns)")
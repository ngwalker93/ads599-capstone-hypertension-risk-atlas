# ==============================================================================
# PROJECT: ADS599 Capstone - Hypertension Risk Atlas
# SCRIPT: 03_get_chrr.R
# AUTHOR: Nancy Walker
# DATE: 2026-07-14
# DESCRIPTION: 
#     The CHR&R repository provides comprehensive county-level health determinants. 
#     This dataset must be placed in the raw data directory manually.
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

### County Health Rankings & Roadmaps (CHR&R)

# Define the exact path where the file must live
chrr_csv <- "data/raw/analytic_data2025.csv"

# Defensive Check: If the file is missing, halt and print instructions
if (!file.exists(chrr_csv)) {
  stop(paste0(
    "\n\n❌ MISSING REQUIRED DATASET!\n",
    "Please follow these steps to add the CHR&R data:\n",
    "1. Go to: https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation \n",
    "2. Under the '2025 Annual Data Release' section, download the '2025 CHR CSV Analytic Data'.\n",
    "3. Move the file into this project folder under: data/raw/\n",
    "4. Rename it exactly to: analytic_data2025.csv\n\n"
  ))
}

# If the file exists, safely read it into the active R session
message("✅ CHR&R dataset found! Ingesting data...")
chrr_raw <- readr::read_csv(chrr_csv, skip = 1, show_col_types = FALSE)

# Extract a small subset to verify data parity
chrr_test <- chrr_raw %>%
  head(5) %>%
  dplyr::select(state, county, v001_rawvalue)

# Render a clean preview table
knitr::kable(chrr_test, caption = "CHR&R Health Policy Registry Sample Data")
# ==============================================================================
# PROJECT: ADS599 Capstone - Hypertension Risk Atlas
# SCRIPT: 02_get_foodacess.R
# AUTHOR: Nancy Walker
# DATE: 2026-07-14
# DESCRIPTION: 
#    The USDA server restricts automated programmatic downloads of census-tract-level food 
#    accessibility metrics from the USDA Economic Research Service. This dataset must be placed 
#    in the raw data directory manually.
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

### SUSDA Food Access Research Atlas

# Define the exact path where the file must live
local_xlsx <- "data/raw/FoodAccessResearchAtlasData2019.xlsx"

# Defensive Check: If the file is missing, halt and print clean instructions
if (!file.exists(local_xlsx)) {
  stop(paste0(
    "\n❌ MISSING REQUIRED DATASET!\n",
    "Please follow these steps to add the USDA Food Atlas data:\n",
    "1. Go to: https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data#:~:text=Food%20Access%20Research%20Atlas%20Data%20Download%202019 \n",
    "2. Click 'Download Food Access Research Atlas Data Download 2019 (XLSX, 81.83 MB)' under the Current Version header.\n",
    "3. Move the downloaded file into this project folder under: data/raw/\n",
    "4. Rename it exactly to: FoodAccessResearchAtlasData2019.xlsx\n\n"
  ))
}

# If the file exists, safely read it into the active R session
message("✅ USDA dataset found! Ingesting data...")
usda_raw <- readxl::read_excel(local_xlsx, sheet = "Food Access Research Atlas")

# Extract a small subset to verify data parity
usda_test <- usda_raw %>%
  head(5) %>%
  dplyr::select(State, County, CensusTract, LowIncomeTracts, LAPOP1_10)

# Render a clean preview table
knitr::kable(usda_test, caption = "USDA Food Access Research Atlas Sample Data")
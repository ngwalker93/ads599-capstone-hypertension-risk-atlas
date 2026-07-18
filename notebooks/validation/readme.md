# EDA Notebooks/Validation — Hypertension Risk Atlas

This folder contains the exploratory data analysis (EDA) notebooks for the
Hypertension Risk Atlas capstone project from 4 different sourcing data.

## Contents

| Notebook | Scope | Status |
|---|---|---|
| `eda.ipynb` | Legacy EDA against the earlier four-source R merge (`master_dataset.csv`, 872 columns, includes CHR&R) | Superseded |
| `eda_master.ipynb` | Current EDA against the Python pipeline's three-source master table (`master_dataset_all_variables.csv`, 120 columns: CDC PLACES + USDA FARA + Census ACS) | Active |

CHR&R is not yet merged into the active master table — its feature set (796
raw columns) requires a separate feature-selection pass before it can be
folded in without overwhelming the model with redundant, highly missing
columns.

## Missing Value Policy

Every numeric column is bucketed into a missingness tier, and each tier maps
to a specific treatment:

| Missing % | Tier | Action |
|---|---|---|
| 0% | Complete | No action |
| < 5% | Minimal | Impute with column median |
| 5–20% | Low | Impute with median (group-wise median where a natural grouping exists, e.g. by state) |
| 20–40% | Moderate | Impute with median; add a `<column>_is_missing` indicator so the model can see that the value was imputed |
| 40–60% | High | Case-by-case review — decide fill vs. drop based on whether the missingness is structural (e.g. a rate that's undefined for some rows) or genuine |
| ≥ 60% | Severe | Drop the column — too little signal remains to justify keeping it, and it inflates multicollinearity risk |

**Current snapshot (`master_dataset_all_variables.csv`, 3,231 rows × 120 columns):**
No column exceeds ~29% missing. The highest-missingness columns are
`LONELINESS`, `FOODSTAMP`, `LACKTRPT`, `HOUSINSECU`, `EMOTIONSPT`, and
`FOODINSECU`, all at 28.85% — comfortably within the "Moderate" tier, so
nothing currently qualifies for the drop threshold. This is expected to
change once CHR&R (much higher missingness, per-column, due to suppression
flags) is merged in.

## Acknowledgements

**Data sources:** CDC PLACES, USDA Food Access Research Atlas, U.S. Census
Bureau (ACS 5-Year Estimates), and County Health Rankings & Roadmaps
(CHR&R) — all public and freely accessible.

**Team:** Nancy Walker & Michelle Wang, ADS-599 Capstone, University of San
Diego.

**AI assistance:** AI tools were used to support drafting and code
scaffolding during the development of these notebooks. All generated output
was reviewed, validated, and edited by the authors, who take full
responsibility for the final content.
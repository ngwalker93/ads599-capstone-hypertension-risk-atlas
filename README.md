# Hypertension Risk Atlas

**ADS-599 Capstone Project — University of San Diego**
**Project Group 2:** Nancy Walker & Michelle Wang

An interpretable machine learning project that predicts county-level hypertension prevalence across the United States from place-based social and environmental determinants, and explains the specific drivers behind each county's risk.

---

## Project Status

**Current stage: Module 1 — Project Proposal (Stage 1: Project Ideation).**
This repository accompanies the Capstone Proposal submission. Team formation, data sourcing, and the initial data integration (ETL) are complete; exploratory analysis, modeling, and the interactive product are planned for upcoming modules.

| Phase | Status |
|-------|--------|
| Team formation & proposal | Complete |
| Data acquisition (4 public sources) | Complete |
| ETL / merge to county-level dataset | Complete |
| Data cleaning & feature engineering | In progress |
| Exploratory data analysis | Planned |
| Modeling & evaluation | Planned |
| Interactive atlas (Streamlit) | Planned |

---

## Problem & Goal

Hypertension affects nearly half of U.S. adults and is unevenly distributed across geography. Public health agencies repeatedly face the same question: where should limited prevention resources be directed to reduce hypertension burden most effectively?

This project reframes that question as a data science problem — predicting county-level hypertension prevalence from social, environmental, and food-access indicators — and uses interpretable machine learning (SHAP) to expose *why* each county is at risk, not just *where* risk is high.

---

## Data Sources

All four sources are public, free, and joined on the 5-digit county FIPS / GEOID.

| Source | Contribution | Size (raw) | Access |
|--------|-------------|------------|--------|
| CDC PLACES | Health outcomes incl. target (hypertension prevalence) | 3,144 counties × 24 cols | Socrata API (RSocrata) / download; no key |
| USDA Food Access Research Atlas | Food environment & access indicators | 72,531 tracts × 147 cols | Direct download; no key |
| U.S. Census Bureau (ACS) | Socioeconomic context (income, poverty, housing) | 6,444 records × 5 cols | Census Data API; free key |
| County Health Rankings (CHR&R) | Social & clinical community health metrics | 3,204 counties × 796 cols | Download / `countyhealthR`; no key |

**Merged analytic dataset:** all sources standardized to county-level GEOID and joined into a single table of **3,231 rows × 120 columns** (before cleaning). Target variable: county-level hypertension prevalence (BPHIGH).

---

## Planned Methodology

1. **Data acquisition** — Programmatic ingestion from the four primary sources (CDC, USDA, Census, and CHR&R), with automated logging of source versions and metadata.
2. **Data preparation** — Standardization to 5-digit fipscode (GEOID) across all datasets, resolution of data-grain differences, and strict isolation of modeling data from validation data. 
3. **Exploratory data analysis** — Assessment of distributions, geographic patterns, correlation heatmaps, and systematic handling of multicollinearity.
4. **Relational Integration** — All datasets are processed through a modular Python pipeline and hosted in a SQLite relational database, enabling reproducible analysis and dashboard-ready SQL views.
5. **Feature Engineering and Dimensionality Reduction** — Creation of new features (e.g., food access ratios, composite socioeconomic indices) and application of PCA to reduce dimensionality while retaining variance in the predictors.
5. **Modeling** — Linear Regression baseline vs. Random Forest and XGBoost.Feature selection via correlation analysis; establishment of Linear Regression baselines compared against ensemble models (Random Forest and XGBoost).
6. **Evaluation** — Performance assessment using $R^2$, RMSE, and MAE against baselines; residual analysis to identify potential geographic bias.
7. **Interpretation** — Utilization of SHAP (SHapley Additive exPlanations) to interpret model predictions and quantify the influence of place-based drivers on hypertension risk.

---

## Planned Repository Structure

```
Hypertension-Risk-Atlas/
├── data/
│   ├── figures/         # exploratory analysis figures (not version-controlled if large)
│   ├── final/           # final analytic dataset (cleaned, merged, and feature-engineered)
│   ├── processed/       # location of all datasets created by the ETL pipeline (cleaned, merged, and feature-engineered)
│   ├── raw/             # source files (not version-controlled if large)
│   └── validation/      # validation datasets (raw source datasets with validation checks)
├── notebooks/
│   ├── DataIngestion/    # folder containing .qmd files that ingest data from CDC, USDA, Census, and CHR&R
│   ├── EDA/              # folder containing .ipynb notebooks for exploratory data analysis
│   └──  validation/      # folder containing .ipynb notebooks for validation of raw source datasets
├── pipeline/
│   ├── DataIngestion/   # ingestion & merge scripts (python)
│   ├── R_src/           # R scripts to retrive raw data via API calls and downloads
│   └── modeling/        # model training & evaluation
├── .gitignore            
├── ADS599_Project.Rproj  # RStudio project file
├── audit_data            # Check for data integrity and completeness
├── hypertension_atlas.db # SQLite database of merged datasets
├── LICENSE 
├── main.py                # main script to run the entire pipeline
├── paths.py              # filepaths for data ingestion and modeling 
├── README.md
├── references.bib         # bibliography for literature review
├── {} renv.lock           # R dependencies
├── requirements.txt       # Python dependencies             
└── utils.py               # utility functions for data ingestion and modeling
```

---

## Planned Deliverables (Module 7)

- **Capstone Article** — full methodology and results
- **Capstone GitHub** — documented, interview-ready repository
- **Capstone Presentation** — narrated pitch
- **Capstone User Tool** — interactive Hypertension Risk Atlas (Streamlit web app)

---

## Tools & Workflow

- **Languages/Environments:** R (ETL), Python (analysis & modeling), Jupyter Notebook, VS Code
- **Version control:** GitHub (with GitHub Projects Kanban board for task tracking)
- **Collaboration:** Slack (coordination), Zoom (working sessions), shared Google Drive (documents)

---

## Activate Virtual Environment (Python)

Use python 3.10+ to create and activate a virtual environment for this project.

```bash
# Create virtual environment (if not already created)
python -m venv .venv
# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

## Install Dependencies (Python)

```bash
# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Install Dependencies (R)

```r
# Install dependencies from renv.lock (if using renv)
# Install renv if not already installed
install.packages("renv")
# Restore packages from renv.lock
renv::restore()
```

---
## Run Data Pipeline

```bash
# Run the main.py script to execute the entire data pipeline
python main.py
```

---

## Scope & Intended Use

The Hypertension Risk Atlas is designed to **identify and prioritize** high-risk counties and inform the planning of targeted prevention activities. Consistent with the intended use of the underlying CDC PLACES small-area estimates, it is **not** intended to evaluate the effectiveness of specific programs or policies.

---

*Note on AI assistance: AI tools were used to support drafting and code scaffolding during this project. During the preparation of this work, the authors used Gemini, an AI-based large language model, to assist with initial research, the synthesis of the literature review, and the development of data ingestion strategies. After using this tool, the authors validated and edited all output content, including technical terminology and research summaries, to ensure accuracy and take full responsibility for the final content of the publication.


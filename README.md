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

**Merged analytic dataset:** all sources standardized to county-level GEOID and joined into a single table of **3,222 rows × 872 columns** (before cleaning). Target variable: county-level hypertension prevalence.

---

## Planned Methodology

1. **Data acquisition** — programmatic ingestion from the four sources (ETL implemented in R), with retrieval date and source version logged for reproducibility.
2. **Data preparation** — standardize to county GEOID, resolve data-grain differences, remove redundant/leakage fields, handle missing values.
3. **Exploratory data analysis** — distributions, geographic patterns, correlations, and multicollinearity.
4. **Modeling** — Lasso/PCA feature pruning; Linear Regression baseline vs. Random Forest and XGBoost.
5. **Evaluation** — R², RMSE, MAE against a baseline; residual analysis for geographic bias.
6. **Interpretation** — SHAP to rank the place-based drivers behind each prediction.

---

## Planned Repository Structure

```
Hypertension-Risk-Atlas/
├── data/
│   ├── raw/             # source files (not version-controlled if large)
│   └── processed/       # master_dataset.csv (cleaned analytic table)
├── code/
│   ├── etl/             # ingestion & merge (R)
│   ├── eda/             # exploratory analysis notebooks
│   └── modeling/        # model training & evaluation
├── images/              # figures, maps, SHAP plots
├── app/                 # Streamlit Hypertension Risk Atlas (planned)
├── docs/                # proposal, capstone article drafts
└── README.md
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

## Scope & Intended Use

The Hypertension Risk Atlas is designed to **identify and prioritize** high-risk counties and inform the planning of targeted prevention activities. Consistent with the intended use of the underlying CDC PLACES small-area estimates, it is **not** intended to evaluate the effectiveness of specific programs or policies.

---

*Note on AI assistance: AI tools were used to support drafting and code scaffolding during this project. During the preparation of this work, the authors used Gemini, an AI-based large language model, to assist with initial research, the synthesis of the literature review, and the development of data ingestion strategies. After using this tool, the authors validated and edited all output content, including technical terminology and research summaries, to ensure accuracy and take full responsibility for the final content of the publication.


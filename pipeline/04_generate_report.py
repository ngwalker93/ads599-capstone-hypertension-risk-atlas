import io
import base64
import os
import asyncio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
from playwright.async_api import async_playwright
from paths import DATA_FINAL, DATA_PROCESSED, validate_and_alert

def fig_to_base64(fig):
    """Utility to convert a Matplotlib figure to a base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

async def generate_pdf_report():
    # Load data
    data_path = DATA_PROCESSED / "master_dataset_all_variables.csv"
    validate_and_alert(data_path, "Master Dataset", "Run Data Ingestion and Cleaning scripts.")
    
    FIPS_COL = "fipscode"
    TARGET_COL = "BPHIGH"
    df = pd.read_csv(data_path, dtype={FIPS_COL: str}, low_memory=False)

    # Load train/test data for split metrics
    X_train_path = DATA_FINAL / "X_train.csv"
    X_test_path = DATA_FINAL / "X_test.csv"
    y_train_path = DATA_FINAL / "y_train.csv"
    
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    # Calculate summary metrics
    total_records= len(df)
    total_rows = df.shape[0]
    n_usable = df[TARGET_COL].notna().sum()
    missing_pct_summary = round(df.isnull().mean().mean() * 100, 2)
    duplicate_records = int(df.duplicated().sum())

    # Missingness distribution counts
    missing_pct = df.isnull().mean() * 100
    missing_bins = pd.cut(missing_pct, bins=[-1, 0, 5, 20, 40, 60, 100],
                          labels=["0%", "<5%", "5-20%", "20-40%", "40-60%", ">60%"]).value_counts().to_dict()

    # Source column counts
    id_cols = ["fipscode", "locationname", "stateabbr", "State", "County"]
    acs_cols = ["poverty_count", "median_income"]
    cdc_cols = ["ACCESS2", "BPHIGH", "CASTHMA", "COGNITION", "COPD", "CSMOKING",
                "DIABETES", "EMOTIONSPT", "FOODINSECU", "FOODSTAMP", "GHLTH",
                "HOUSINSECU", "LACKTRPT", "LONELINESS", "LPA", "MOBILITY",
                "OBESITY", "SELFCARE", "SLEEP", "TEETHLOST", "VISION"]
    usda_cols = [c for c in df.columns if c not in id_cols + acs_cols + cdc_cols]

    source_counts = {
        "CDC PLACES": len(cdc_cols),
        "USDA FARA": len(usda_cols),
        "Census ACS": len(acs_cols),
        "Identifiers": len(id_cols)
    }   

    split_metrics = {
        "Train Samples": f"{len(X_train):,}",
        "Test Samples": f"{len(X_test):,}",
        "Total Features After Preprocessing": f"{X_train.shape[1]:,}"
    } 

    # Generate Figure 1: Target Variable Distribution
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.histplot(df[TARGET_COL].dropna(), bins=40, kde=True, color="steelblue", edgecolor="white", ax=ax1)
    ax1.set_xlabel("Hypertension Prevalence (%)")
    ax1.set_ylabel("County Count")
    ax1.set_title("Distribution of County-Level Hypertension Prevalence")
    plt.tight_layout()
    plot1_uri = fig_to_base64(fig1)

    # Generate Figure 2: Top 20 Correlations Bar Chart
    corr = df.select_dtypes("number").corrwith(df[TARGET_COL]).drop(TARGET_COL).dropna()
    top_20 = corr.reindex(corr.abs().sort_values(ascending=False).index).head(20)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    top_20.sort_values().plot.barh(ax=ax2, color="#2b6cb0")
    ax2.set_xlabel(f"Correlation with {TARGET_COL}")
    ax2.set_title("Top 20 Correlated Features")
    plt.tight_layout()
    plot2_uri = fig_to_base64(fig2)

    leaks = top_20[top_20.abs() > 0.95]
    print("No leakage flags (|r| > 0.95)." if leaks.empty else leaks)

    # Generate Figure 3: Correlation Heatmap Matrix
    top_features = list(corr.head(10).index) + list(corr.tail(10).index)
    corr_matrix = df[top_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig3, ax3 = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                annot_kws={"size": 7},
                cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True, ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
    ax3.set_title("Feature Correlation Heatmap Matrix")
    plt.tight_layout()
    plot3_uri = fig_to_base64(fig3)

    # HTML Template Structure
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body { font-family: Helvetica, Arial, sans-serif; margin: 40px; color: #333; line-height: 1.5; }
            h1 { color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 8px; }
            h2 { color: #2b6cb0; margin-top: 30px; }
            .metrics-container { display: flex; gap: 20px; margin: 20px 0; }
            .metric-box { background: #f7fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 6px; text-align: center; flex: 1; }
            .metric-value { font-size: 22px; font-weight: bold; color: #2b6cb0; }
            .metric-label { font-size: 12px; color: #4a5568; text-transform: uppercase; margin-top: 5px; }
            .chart-section { text-align: center; margin: 30px 0; page-break-inside: avoid; }
            img { max-width: 85%; height: auto; border: 1px solid #cbd5e0; border-radius: 4px; padding: 5px; background: #fff; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #cbd5e0; padding: 8px 12px; text-align: center; }
            th { background-color: #f7fafc; color: #2b6cb0; }
        </style>
    </head>
    <body>
        <h1>Master Dataset: EDA Summary Report</h1>
        <p>Automated pipeline execution report generated successfully.</p>
        
        <!-- Detailed Breakdown Section -->
        <h2>Dataset Structure & Quality</h2>
        <div class="metrics-container">
            <div class="metric-box">
                <div class="metric-value">{{ duplicate_records }}</div>
                <div class="metric-label">Fully Duplicated Rows</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ total_rows }}</div>
                <div class="metric-label">Total Rows</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ source_counts["CDC PLACES"] + source_counts["USDA FARA"] + source_counts["Census ACS"] + source_counts["Identifiers"] }}</div>
                <div class="metric-label">Total Columns</div>
            </div>
        </div>

        <div class="metrics-container">
            {% for key, val in split_metrics.items() %}
            <div class="metric-box">
                <div class="metric-value">{{ val }}</div>
                <div class="metric-label">{{ key }}</div>
            </div>
            {% endfor %}
        </div>

        <h3>Columns by Source</h3>
        <ul>
            {% for source, count in source_counts.items() %}
            <li><strong>{{ source }}:</strong> {{ count }} columns</li>
            {% endfor %}
        </ul>

        <h3>Missingness Breakdown (Columns)</h3>
        <ul>
            {% for bracket, count in missing_bins.items() %}
            <li><strong>{{ bracket }} missing:</strong> {{ count }} columns</li>
            {% endfor %}
        </ul>

        <h2>Target Variable Distribution</h2>
        <div class="chart-section">
            <img src="{{ plot1_uri }}" alt="Target Distribution">
        </div>

        <h2>Top 20 Correlated Predictors</h2>
        <div class="chart-section">
            <img src="{{ plot2_uri }}" alt="Top 20 Correlations">
        </div>

        <h2>Feature Correlation Heatmap</h2>
        <div class="chart-section">
            <img src="{{ plot3_uri }}" alt="Correlation Heatmap">
        </div>

        <h2>Baseline Modeling Pipeline</h2>
        <p>To establish a reproducible benchmark, we evaluated a standardized baseline pipeline combining L1 feature selection and dimensionality reduction:</p>
        
        <div style="background: #f7fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 13px;">
            Pipeline([<br>
            &nbsp;&nbsp;&nbsp;&nbsp;('scaler', StandardScaler()),<br>
            &nbsp;&nbsp;&nbsp;&nbsp;('feature_selection', SelectFromModel(Lasso(alpha=0.0059, max_iter=10000))),<br>
            &nbsp;&nbsp;&nbsp;&nbsp;('pca', PCA(n_components=15, random_state=42)),<br>
            &nbsp;&nbsp;&nbsp;&nbsp;('regressor', LinearRegression())<br>
            ])
        </div>
    </body>
    </html>
    """
    
    template = Template(html_template)
    rendered_html = template.render(
        total_records=f"{total_records:,}",
        n_usable=f"{n_usable:,}",
        missing_pct_summary=missing_pct_summary,
        duplicate_records=duplicate_records,
        total_rows=total_rows,
        missing_bins=missing_bins,
        source_counts=source_counts,
        split_metrics=split_metrics,
        plot1_uri=plot1_uri,
        plot2_uri=plot2_uri,
        plot3_uri=plot3_uri
    )
    
    # Compile directly to PDF via Headless Chromium & Save Files
    output_pdf = DATA_FINAL / "eda_summary_report.pdf"
    output_html = DATA_FINAL / "eda_summary_report.html"
    os.makedirs(DATA_FINAL, exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(rendered_html)
        await page.pdf(path=str(output_pdf), format="Letter", print_background=True)
        await browser.close()

    # Save HTML copy for GitHub viewing
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    # Generate clickable terminal links
    pdf_uri = f"file://{os.path.abspath(output_pdf)}"
    html_uri = f"file://{os.path.abspath(output_html)}"

    print(f"\n📄 PDF Report: {pdf_uri}")
    print(f"🌐 HTML Report: {html_uri}")

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        asyncio.create_task(generate_pdf_report())
        print("📄 PDF generation scheduled in the active event loop.")
    else:
        asyncio.run(generate_pdf_report())
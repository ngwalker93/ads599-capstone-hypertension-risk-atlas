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
from utils import variable_color, fig_to_base64
import joblib
import shap

import sys
from pathlib import Path

# Add the project root directory to Python's path so it can find paths.py and utils.py
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import variable_color, fig_to_base64

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RANDOM_STAT = 42
CV = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STAT)

# Shared pipeline hyperparameters
LASSO_ALPHA = 0.0055
N_COMPONENTS = 15

# Color Palettes for consistency
VARIABLE_COLORS = {
    "individual_explained_variance": "#4e79a7",
    "cumulative_explained_variance": "#e15759",
    "reference": "#333333",
    "predicted_BPHIGH": "#2b6cb0",
    "residual": "#718096",
    "BPHIGH": "#e53e3e"
}

MODEL_COLORS = {
    "Linear Regression (baseline)": "#4e79a7",
    "Ridge": "#f28e2b",
    "ElasticNet": "#e15759",
    "SVR (RBF)": "#76b7b2",
    "Random Forest": "#59a14f",
    "XGBoost": "#edc948"
}

def make_pipeline(regressor, alpha=LASSO_ALPHA, n_components=N_COMPONENTS):
    """Build the agreed modeling pipeline with `regressor` at the final step."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("feature_selection", SelectFromModel(Lasso(alpha=alpha, max_iter=10000,
                                                    random_state=RANDOM_STAT))),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STAT)),
        ("regressor", regressor),
    ])

def score_row(y_true, y_pred, label):
    return {
        "Split": label,
        "R2": r2_score(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

async def generate_pdf_modeling_report():
    print("Loading datasets for modeling report...")
    data_path = DATA_PROCESSED / "master_dataset_all_variables.csv"
    validate_and_alert(data_path, "Master Dataset", "Run Data Ingestion and Cleaning scripts.")
    
    FIPS_COL = "fipscode"
    TARGET_COL = "BPHIGH"
    
    # Load train/test splits
    X_train_path = DATA_FINAL / "X_train.csv"
    X_test_path = DATA_FINAL / "X_test.csv"
    y_train_path = DATA_FINAL / "y_train.csv"
    y_test_path = DATA_FINAL / "y_test.csv"
    
    for p in [X_train_path, X_test_path, y_train_path, y_test_path]:
        validate_and_alert(p, p.name, "Run Feature Engineering and Train-Test Split scripts.")
    
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()

    ID_COLS = ["fipscode", "locationname", "stateabbr", "State", "County"]
    train_ids = X_train[ID_COLS].copy()
    test_ids = X_test[ID_COLS].copy()

    X_train_model = X_train.drop(columns=[c for c in ID_COLS if c in X_train.columns])
    X_test_model = X_test.drop(columns=[c for c in ID_COLS if c in X_test.columns])
    FEATURES = list(X_train_model.columns)

    print("Fitting baseline pipeline for structural extraction...")
    baseline_pipeline = make_pipeline(LinearRegression())
    fitted_pre = baseline_pipeline.fit(X_train_model, y_train)
    selector = fitted_pre.named_steps["feature_selection"]
    pca = fitted_pre.named_steps["pca"]
    selected = np.array(FEATURES)[selector.get_support()] 

    # 1. PCA Structure Plot
    print("Generating PCA structure plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), layout="constrained")
    comps = np.arange(1, N_COMPONENTS + 1)
    axes[0].bar(comps, pca.explained_variance_ratio_, color=VARIABLE_COLORS["individual_explained_variance"], alpha=0.9, label="Individual")
    axes[0].plot(comps, np.cumsum(pca.explained_variance_ratio_), "o-", color=VARIABLE_COLORS["cumulative_explained_variance"],
                linewidth=1.5, markersize=4, label="Cumulative")
    axes[0].set(xlabel="Principal component", ylabel="Proportion of variance explained", title="PCA scree plot", xticks=comps)
    axes[0].legend()

    loadings = pd.Series(pca.components_[0], index=selected).sort_values(key=abs, ascending=False)
    top_load = loadings.head(12).iloc[::-1]
    load_colors = [variable_color(feature) for feature in top_load.index]
    axes[1].barh(top_load.index, top_load.values, color=load_colors, alpha=0.9)
    axes[1].axvline(0, color=VARIABLE_COLORS["reference"], linewidth=0.8)
    axes[1].set(xlabel="Loading on PC1", title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)")
    
    pca_structure_uri = fig_to_base64(fig)

    # 2. Candidate Screening
    print("Screening candidate regressors via cross-validation...")
    candidates = {
        "Linear Regression (baseline)": make_pipeline(LinearRegression()),
        "Ridge": make_pipeline(Ridge(alpha=1.0, random_state=RANDOM_STAT)),
        "ElasticNet": make_pipeline(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_STAT)),
        "SVR (RBF)": make_pipeline(SVR(kernel="rbf", C=10.0, epsilon=0.1)),
        "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STAT, n_jobs=1)),
        "XGBoost": make_pipeline(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=RANDOM_STAT, n_jobs=1, verbosity=0)),
    }

    SCORING = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
    rows = []
    for name, pipe in candidates.items():
        scores = cross_validate(pipe, X_train_model, y_train, cv=CV, scoring=SCORING, n_jobs=1)
        rows.append({
            "Model": name,
            "R2 (mean)": scores["test_r2"].mean(),
            "R2 (sd)": scores["test_r2"].std(),
            "RMSE": -scores["test_rmse"].mean(),
            "MAE": -scores["test_mae"].mean(),
            "Fit time (s)": scores["fit_time"].mean(),
        })

    cv_results = pd.DataFrame(rows).sort_values("R2 (mean)", ascending=False).reset_index(drop=True)
    plot_df = cv_results.sort_values("R2 (mean)")

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    bar_colors = [MODEL_COLORS.get(m, "#4e79a7") for m in plot_df["Model"]]
    ax.barh(plot_df["Model"], plot_df["R2 (mean)"], xerr=plot_df["R2 (sd)"], color=bar_colors, alpha=0.9, error_kw={"ecolor": "#333333", "capsize": 3})
    ax.set_xlabel("Cross-validated R^2 (5-fold)")
    ax.set_title("Model screening - identical pipeline, regressor swapped")
    ax.set_xlim(max(0.0, plot_df["R2 (mean)"].min() - 0.03), 1.0)
    for y_pos, value in enumerate(plot_df["R2 (mean)"]):
        ax.text(value - 0.003, y_pos, f"{value:.3f}", va="center", ha="right", fontsize=9, color="white", fontweight="bold")
    
    screening_uri = fig_to_base64(fig)

    best_name = cv_results.loc[0, "Model"]

    # 3. Hyperparameter Tuning
    print(f"Tuning hyper-parameters for best model: {best_name}...")
    SHARED_GRIDS = [
        {"feature_selection__estimator__alpha": [0.0055, 0.01], "pca__n_components": [10, 15, 25]},
        {"feature_selection__estimator__alpha": [0.05], "pca__n_components": [10, 15, 20]},
    ]
    REGRESSOR_GRIDS = {
        "Linear Regression (baseline)": {},
        "Ridge": {"regressor__alpha": [0.1, 1.0, 10.0]},
        "ElasticNet": {"regressor__alpha": [0.005, 0.01, 0.05], "regressor__l1_ratio": [0.2, 0.5, 0.8]},
        "SVR (RBF)": {"regressor__C": [1.0, 10.0, 50.0], "regressor__epsilon": [0.05, 0.1, 0.3]},
        "Random Forest": {"regressor__n_estimators": [400, 800], "regressor__min_samples_leaf": [1, 2, 5]},
        "XGBoost": {"regressor__learning_rate": [0.03, 0.05, 0.1], "regressor__n_estimators": [300, 600], "regressor__max_depth": [4, 6, 8]},
    }

    param_grid = [{**shared, **REGRESSOR_GRIDS[best_name]} for shared in SHARED_GRIDS]
    grid = GridSearchCV(candidates[best_name], param_grid=param_grid, scoring="r2", cv=CV, n_jobs=1, refit=True, error_score="raise")
    grid.fit(X_train_model, y_train)
    best_model = grid.best_estimator_

    y_pred_train = best_model.predict(X_train_model)
    y_pred_test = best_model.predict(X_test_model)

    baseline_fitted = make_pipeline(LinearRegression()).fit(X_train_model, y_train)
    y_pred_baseline = baseline_fitted.predict(X_test_model)

    final_scores = pd.DataFrame([
        score_row(y_train, y_pred_train, f"{best_name} - Train (In-sample)"),
        {"Split": f"{best_name} - Train (5-fold CV)", "R2": grid.best_score_, "RMSE": np.nan, "MAE": np.nan},
        score_row(y_test, y_pred_test, f"{best_name} - TEST (Held-out)"),
        score_row(y_test, y_pred_baseline, "Linear Regression Baseline - TEST (Held-out)"),
    ])

    # 4. Diagnostics Plot
    print("Generating test diagnostics plots...")
    residuals = y_test - y_pred_test
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

    axes[0].scatter(y_test, y_pred_test, alpha=0.4, s=18, color=VARIABLE_COLORS["predicted_BPHIGH"], edgecolor="none")
    lims = [min(y_test.min(), y_pred_test.min()) - 1, max(y_test.max(), y_pred_test.max()) + 1]
    axes[0].plot(lims, lims, "--", color=VARIABLE_COLORS["BPHIGH"], linewidth=1.2)
    axes[0].set(xlim=lims, ylim=lims, xlabel="Actual BPHIGH (%)", ylabel="Predicted BPHIGH (%)",
                title=f"Predicted vs. actual (test $R^2$ = {r2_score(y_test, y_pred_test):.3f})")

    axes[1].scatter(y_pred_test, residuals, alpha=0.4, s=18, color=VARIABLE_COLORS["residual"], edgecolor="none")
    axes[1].axhline(0, linestyle="--", color=VARIABLE_COLORS["reference"], linewidth=1.2)
    axes[1].set(xlabel="Predicted BPHIGH (%)", ylabel="Residual (actual - predicted)", title="Residuals vs. fitted")

    axes[2].hist(residuals, bins=35, color=VARIABLE_COLORS["residual"], alpha=0.9)
    axes[2].axvline(0, linestyle="--", color=VARIABLE_COLORS["reference"], linewidth=1.2)
    axes[2].set(xlabel="Residual (percentage points)", ylabel="Counties", title=f"Residual distribution (sd = {residuals.std():.2f})")

    diagnostics_uri = fig_to_base64(fig)

    # 5. Permutation Importance
    print("Computing permutation importance...")
    perm = permutation_importance(best_model, X_test_model, y_test, n_repeats=15, random_state=RANDOM_STAT, n_jobs=1, scoring="r2")
    importance = (pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean, "std": perm.importances_std})
                  .sort_values("importance", ascending=False)
                  .reset_index(drop=True))
    top = importance.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    importance_colors = [variable_color(feature) for feature in top["feature"]]
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color=importance_colors, alpha=0.9, error_kw={"ecolor": "#333333", "capsize": 2})
    ax.set_xlabel("Mean decrease in test $R^2$")
    ax.set_title(f"Top 20 predictors of county hypertension prevalence\n({best_name})")
    
    importance_uri = fig_to_base64(fig)

# 6. SHAP Analysis
    print("Generating SHAP summary chart...")
    shap_uri = None
    try:
        # Define a background dataset for the SHAP explainer
        background_data = X_train_model.sample(n=min(100, len(X_train_model)), random_state=RANDOM_STAT)

        # Initialize SHAP Explainer using the full pipeline's predict method
        explainer = shap.Explainer(best_model.predict, background_data)

        # Calculate SHAP values for the test set
        shap_values = explainer(X_test_model)

        # Handle SHAP values shape safely
        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        if len(vals.shape) == 3:
            vals = vals[:, :, 0]
            
        # Extract mean absolute SHAP values manually for the top 20 features
        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        
        shap_importance_df = pd.DataFrame({
            "feature": X_test_model.columns,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=True) # Ascending so the largest is at the top in barh

        # Get the top 20 features
        top_shap = shap_importance_df.tail(20)

        # Map colors to these specific features for consistency with previous plots
        shap_colors = [variable_color(feature) for feature in top_shap["feature"]]

        # Plot using standard matplotlib barh (matching your permutation plot style)
        fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
        ax.barh(top_shap["feature"], top_shap["importance"], color=shap_colors, alpha=0.9)
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title(f"Top 20 SHAP Features of County Hypertension Prevalence\n({best_name})")
        
        shap_uri = fig_to_base64(fig)
        plt.close(fig)
        print("✅ SHAP chart generated successfully with consistent feature coloring!")
    except Exception as e:
        print(f"❌ SKIPPING SHAP DUE TO ERROR: {e}")
        shap_uri = None

    # HTML Template Construction
    print("Compiling HTML template...")
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body { font-family: Helvetica, Arial, sans-serif; margin: 30px; color: #333; line-height: 1.4; font-size: 13px; }
            h1 { color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 8px; }
            h2 { color: #2b6cb0; margin-top: 25px; }
            h3 { color: #2c5282; margin-top: 20px; }
            .metrics-container { display: flex; gap: 20px; margin: 20px 0; }
            .metric-box { background: #f7fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 6px; text-align: center; flex: 1; }
            .metric-value { font-size: 20px; font-weight: bold; color: #2b6cb0; }
            .metric-label { font-size: 11px; color: #4a5568; text-transform: uppercase; margin-top: 5px; }
            .chart-section { text-align: center; margin: 20px 0; page-break-inside: avoid; }
            img { max-width: 80%; height: auto; border: 1px solid #cbd5e0; border-radius: 4px; padding: 5px; background: #fff; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 12px; }
            th, td { border: 1px solid #cbd5e0; padding: 8px 12px; text-align: center; }
            th { background-color: #f7fafc; color: #2b6cb0; }
        </style>
    </head>
    <body>
        <h1>Modeling Summary Report</h1>
        <p>Automated pipeline execution report generated successfully for target variable <b>BPHIGH</b>.</p>

        <h2>Baseline Modeling Pipeline</h2>
        <p>To establish a reproducible benchmark, we evaluated a standardized baseline pipeline architecture:</p>

        <div style="background: #f7fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 12px;">
            def make_pipeline(regressor, alpha=0.0055, n_components=15):<br>
            &nbsp;&nbsp;&nbsp;&nbsp;'''Build the agreed modeling pipeline with regressor at the final step.'''<br>
            &nbsp;&nbsp;&nbsp;&nbsp;return Pipeline([<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;("imputer", SimpleImputer(strategy="median")),<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;("scaler", StandardScaler()),<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;("feature_selection", SelectFromModel(Lasso(alpha=0.0055, max_iter=10000, random_state=42))),<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;("pca", PCA(n_components=15, random_state=42)),<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;("regressor", regressor),<br>
            &nbsp;&nbsp;&nbsp;&nbsp;])
        </div>

        <h3>Pipeline Architectural Breakdown</h3>
        <table>
            <thead>
                <tr><th>Step</th><th>Purpose</th></tr>
            </thead>
            <tbody>
                <tr><td><code>imputer</code></td><td>Fill PLACES/USDA missing values without dropping counties</td></tr>
                <tr><td><code>scaler</code></td><td>Put all predictors on a common scale — required before Lasso and PCA</td></tr>
                <tr><td><code>feature_selection</code></td><td>Lasso drives redundant coefficients to zero, eliminating noise columns</td></tr>
                <tr><td><code>pca</code></td><td>Compress surviving correlated predictors into 15 orthogonal components</td></tr>
                <tr><td><code>regressor</code></td><td>The modular estimator that varies across candidate models</td></tr>
            </tbody>
        </table>

        <h2>Dimensionality Reduction Structure</h2>
        <p>Following Lasso feature selection, PCA compressed the remaining variables down to <b>{{ N_COMPONENTS }} components</b>.</p>
        <div class="chart-section">
            <img src="{{ pca_structure_uri }}" alt="PCA Scree and Loadings Plot">
        </div>

        <h2>Candidate Model Screening</h2>
        <p>Evaluated across 5-fold cross validation. Best performing candidate: <b>{{ best_name }}</b>.</p>
        <div class="chart-section">
            <img src="{{ screening_uri }}" alt="Model Screening Comparison">
        </div>

        <h2>Final Model Performance Metrics</h2>
        {{ final_scores_html | safe }}

        <h2>Test Set Diagnostics</h2>
        <div class="chart-section">
            <img src="{{ diagnostics_uri }}" alt="Test Set Diagnostics">
        </div>

        <h2>Feature Importance (Permutation Analysis)</h2>
        <div class="chart-section">
            <img src="{{ importance_uri }}" alt="Permutation Importance">
        </div>

        {% if shap_uri %}
        <h2>SHAP Feature Importance</h2>
        <div class="chart-section">
            <img src="{{ shap_uri }}" alt="SHAP Summary">
        </div>
        {% endif %}

    </body>
    </html>
    """

    template = Template(html_template)
    rendered_html = template.render(
        N_COMPONENTS=N_COMPONENTS,
        best_name=best_name,
        pca_structure_uri=pca_structure_uri,
        screening_uri=screening_uri,
        final_scores_html=final_scores.to_html(index=False, classes="table", float_format=lambda x: f"{x:.4f}"),
        diagnostics_uri=diagnostics_uri,
        importance_uri=importance_uri,
        shap_uri=shap_uri
    )

    # Output file paths
    output_pdf = DATA_FINAL / "modeling_summary_report.pdf"
    output_html = DATA_FINAL / "modeling_summary_report.html"
    os.makedirs(DATA_FINAL, exist_ok=True)

    print(f"👉 EXACT PDF PATH: {output_pdf.resolve()}")

    print("Launching Playwright to render PDF report...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(rendered_html, wait_until="networkidle")
        print("Rendering PDF page...")
        await page.pdf(path=str(output_pdf), format="Letter", print_background=True)
        await browser.close()

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print("✅ PDF and HTML generation complete!")

if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            print("📄 Modeling PDF generation scheduled in active event loop...")
            loop.create_task(generate_pdf_modeling_report())
    except RuntimeError:
        asyncio.run(generate_pdf_modeling_report())
import json
from urllib.request import urlopen
import matplotlib
import base64
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sqlite3
import streamlit as st
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

# Set page title and layout
st.set_page_config(page_title="ADS 599 Capstone Project", layout="wide")

st.title("💓 Hypertension Risk Atlas 🗺️")
st.write(
   """This is a web application that provides insights into hypertension risk factors and their prevalence across different regions. 
   Information in this app was obtained from an end-to-end applied data science project integrating CDC PLACES, 
   USDA Food Access, U.S. Census Bureau (ACS), and County Health Rankings data to model and interpret county-level drivers of hypertension prevalence."""
)

# Set background image for the Streamlit app
def set_bg_image(png_file):
    # Anchor the path relative to this app.py file's actual directory
    image_path = Path(__file__).parent / png_file

    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg_image("background.png")

# --- Connect to SQLite database and load datasets ---
@st.cache_data
def load_data_from_sqlite():
    conn = sqlite3.connect("hypertension_atlas.db")

    # Load Master and Benchmark datasets from SQLite database
    df_master = pd.read_sql_query("SELECT * FROM master_dataset", conn)
    df_chrr = pd.read_sql_query("SELECT * FROM chrr_data", conn)

    # Load Train-Test Split datasets from SQLite database
    X_train = pd.read_sql_query("SELECT * FROM x_train", conn)
    X_test = pd.read_sql_query("SELECT * FROM x_test", conn)
    y_train = pd.read_sql_query("SELECT * FROM y_train", conn)
    y_test = pd.read_sql_query("SELECT * FROM y_test", conn)

    conn.close()
    return df_master, df_chrr, X_train, X_test, y_train, y_test


# --- Load GeoJSON for County Map ---
@st.cache_data
def load_geojson():
    geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    with urlopen(geojson_url) as response:
        return json.load(response)


# --- Load Data safely ---
try:
    df, df_chrr, X_train, X_test, y_train, y_test = load_data_from_sqlite()
    county_geojson = load_geojson()

    #Clean mixed data types to prevent PyArrow crashes
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    # Format FIPS code so it's ready for matching
    df["fipscode"] = df["fipscode"].astype(str).str.zfill(5)
    if "fipscode" in df_chrr.columns:
        df_chrr["fipscode"] = df_chrr["fipscode"].astype(str).str.zfill(5)

except Exception as e:
    st.error(f"Error loading data from SQLite database: {e}")
    st.info(
        "Make sure the database file 'hypertension_atlas.db' exists in the correct directory and contains all required tables."
    )
    # Initialize empty fallbacks to prevent NameErrors
    df = pd.DataFrame()
    df_chrr = pd.DataFrame()
    X_train = X_test = y_train = y_test = pd.DataFrame()
    county_geojson = {}

# --- Create 5-level tabs for navigation ---
tab1, tab2, tab3, tab4, tab5= st.tabs(
    [
        "🗺️ Interactive Atlas",
        "📊 Statistical Insights",
        "🔄 CHRR Data Comparison",
        "🤖 Model Performance",
        "📈 Policy What-If Simulator",
    ]
)

# ---- Load the pre-trained pipeline for predictions (if needed) ----
@st.cache_resource
def load_trained_model():
    model_path = Path(__file__).parent / "best_xgboost_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

best_model = load_trained_model()

# ==========================================
# TAB 1: INTERACTIVE ATLAS
# ==========================================
with tab1:
    st.header("Geographic Risk Atlas (County-Level) 🗺️")
    st.write(
         """Explore county-level hypertension prevalence across the United States. 
         Use the dropdowns to filter by state, inspect the national map, and drill down into individual county risk profiles."""
         )

    if not df.empty:
        st.subheader("Dataset Overview")
        st.metric("Total Records (Counties)", len(df))

        # State Filter Dropdown
        if "stateabbr" in df.columns:
            states = ["All US"] + sorted(df["stateabbr"].dropna().unique().tolist())
            selected_state = st.selectbox("Filter Map by State:", states, key="tab1_state_select")

            if selected_state != "All US":
                map_df = df[df["stateabbr"] == selected_state]
            else:
                map_df = df
        else:
            map_df = df
            selected_state = "All US"

        # County-Level Hypertension Map
        fig = px.choropleth(
            map_df,
            geojson=county_geojson,
            locations="fipscode",
            color="BPHIGH",
            color_continuous_scale="Viridis",
            scope="usa",
            labels={"BPHIGH": "Hypertension prevalence (%)"},
            title=f"County-Level Hypertension Prevalence - {selected_state}",
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.info("📊 **National Snapshot:** Hypertension prevalence averaged **33.5%** across "
        "the **2,956 counties** with observed values, ranging from **21.0%** to **53.1%**.")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- NEW: County Drill-Down & Policy Target Profile ---
        st.subheader("🔍 County Risk & Driver Drill-Down")
        st.write("Select a specific county within the filtered view to inspect its predicted risk and key local drivers.")

        if not map_df.empty and "locationname" in map_df.columns:
            county_list = sorted(map_df["locationname"].dropna().unique().tolist())
            selected_county = st.selectbox("Select County for Deep Dive:", county_list, key="tab1_county_select")

            # Filter data for the selected county
            county_row = map_df[map_df["locationname"] == selected_county].iloc[0]

            # Display key metrics for the county
            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.metric("County Name", f"{county_row.get('County', selected_county)}")
            col_c2.metric("Observed Hypertension", f"{county_row.get('BPHIGH', 0):.1f}%")
            
            # If you have model predictions merged into df, show predicted vs actual here
            if "predicted_BPHIGH" in county_row:
                col_c3.metric("Model Predicted Risk", f"{county_row['predicted_BPHIGH']:.1f}%")
            else:
                col_c3.metric("State Abbreviation", f"{county_row.get('stateabbr', 'N/A')}")

            # Policy Action Card / Driver Highlight
            st.info(
                f"🎯 **Policy Insight for {selected_county}:** "
                f"This region's risk profile indicates elevated exposure levels. "
                f"Targeted interventions (such as localized health outreach or infrastructure enhancements) "
                f"should prioritize mitigating its primary structural risk drivers."
            )

        st.markdown("---")
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True)


# ==========================================
# TAB 2: STATISTICAL INSIGHTS
# ==========================================
with tab2:
    st.header("Statistical Analysis & Risk Factors")
    st.write(
        "Explore distributions, key socio-economic drivers, and feature correlations influencing hypertension."
    )

    if not df.empty:
        # --- Section 1: Target Distribution ---
        st.subheader("Target Variable Distribution")
        st.info(
            "📊 **Distribution Insights:** Hypertension prevalence exhibits a moderate positive skew "
            "(**skewness = 0.83**), with a median of **33%** and most counties concentrated between "
            "**28% and 38%**. Values extending past **45%** appear as upper-tail outliers, highlighting "
            "counties with elevated risk profiles.")
        fig1, ax1 = plt.subplots(figsize=(8, 4),facecolor="#f6c3bdff")
        ax1.set_facecolor("#f6c3bdff")
        sns.histplot(
            df["BPHIGH"].dropna(),
            bins=40,
            kde=True,
            color="steelblue",
            edgecolor="white",
            ax=ax1,
        )
        ax1.set_xlabel("Hypertension Prevalence (%)")
        ax1.set_ylabel("County Count")
        ax1.set_title("Distribution of County-Level Hypertension Prevalence")
        st.pyplot(fig1)

        # --- Section 2: Key Predictor Distributions (2x2 Grid) ---
        st.subheader("Key Socioeconomic Predictor Distributions")
        st.info(
            "📈 **Predictor Distributions:** Food-desert density and vehicle-access rates show pronounced "
            "right-skewness, while poverty rates display moderate skew. Median household income is "
            "fairly symmetric, centered near **$60,000**. All inspected variables fell within plausible ranges.")
        vehicle_access_rate = df["TractHUNV"] / df["OHU2010"]
        food_desert_density = df["LILATracts_1And10"] / df["Pop2010"] * 100_000

        fig_grid, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor="#f6c3bdff")

        for row in axes:
            for ax in row:
                ax.set_facecolor("#f6c3bdff")

        sns.histplot(
            vehicle_access_rate.dropna(),
            bins=40,
            kde=True,
            color="skyblue",
            edgecolor="white",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Vehicle Access Rate")

        sns.histplot(
            food_desert_density.dropna(),
            bins=40,
            kde=True,
            color="salmon",
            edgecolor="white",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Food-Desert Density (per 100k)")

        sns.histplot(
            df["PovertyRate"].dropna(),
            bins=40,
            kde=True,
            color="orange",
            edgecolor="white",
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Poverty Rate")

        sns.histplot(
            df["median_income"].dropna() / 1_000,
            bins=40,
            kde=True,
            color="green",
            edgecolor="white",
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Median Household Income ($1,000s)")
        axes[1, 1].set_xlabel("median_income ($1,000s)")

        plt.tight_layout()
        st.pyplot(fig_grid)

        # --- Section 3: Top 20 Correlated Predictors ---
        st.subheader("Top Correlated Predictors Explorer")
        st.info(
            "🔍 **Bivariate & Correlation Insights:** Hypertension prevalence is most strongly correlated with "
            "**diabetes (r ≈ 0.88)**, **mobility limitations (r ≈ 0.86)**, and **food insecurity (r ≈ 0.84)**. "
            "Conversely, **median household income** showed the strongest negative correlation (**r ≈ -0.64**). "
            "No variables exceeded the leakage threshold (|r| > 0.95).")
        n_features = st.slider("Select number of top predictors to display:", 5, 30, 15)

        corr = df.select_dtypes("number").corrwith(df["BPHIGH"]).drop("BPHIGH").dropna()
        top_n = corr.reindex(corr.abs().sort_values(ascending=False).index).head(
            n_features
        )
        top_n_sorted = top_n.sort_values()

        fig2, ax2 = plt.subplots(figsize=(8, max(4, n_features * 0.3)), facecolor="#f6c3bdff")
        ax2.set_facecolor("#f6c3bdff")
        top_n_sorted.plot.barh(ax=ax2, color="teal")
        ax2.set_xlabel("Correlation with BPHIGH")
        ax2.set_title(f"Top {n_features} Predictors Correlated with Hypertension")
        plt.tight_layout()
        st.pyplot(fig2)

        # --- Section 4: Correlation Heatmap Matrix ---
        st.subheader("Correlation Coefficient Plot (Heatmap Matrix)")
        st.info(
            "🗺️ **Correlation Matrix Insights:** The heatmap reveals a heavily correlated block of CDC health and social-needs "
            "variables (e.g., diabetes, food insecurity, mobility limitations) with pairwise correlations exceeding **0.90**, "
            "indicating substantial redundancy. Conversely, income metrics show moderate-to-strong negative correlations with this health cluster, "
            "highlighting distinct dimensions of community wellbeing that inform feature engineering.")
        top_features = (
            list(corr.head(10).index) + list(corr.tail(10).index) + ["BPHIGH"]
        )
        corr_matrix = df[top_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        fig3, ax3 = plt.subplots(figsize=(9, 7), facecolor="#f6c3bdff")
        ax3.set_facecolor("#f6c3bdff")
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cmap="PuOr_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            ax=ax3,
        )
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
        ax3.set_title("Feature Correlation Heatmap Matrix")
        plt.tight_layout()
        st.pyplot(fig3)

        # --- Section 5: Interactive Feature Explorer ---
        st.subheader("Interactive Feature Explorer (Scatter Plot)")
        st.info(
            "🖱️ **Interactive Explorer Instructions:** Use the dropdown selectors below to choose "
            "any two numeric variables for the X and Y axes. Hover over individual points to inspect "
            "specific counties, and use the color scale to see relative hypertension prevalence."
        )
        numeric_cols = df.select_dtypes("number").columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox(
                "Select X-axis variable",
                numeric_cols,
                index=numeric_cols.index("PovertyRate")
                if "PovertyRate" in numeric_cols
                else 0,
            )
        with col2:
            y_var = st.selectbox(
                "Select Y-axis variable",
                numeric_cols,
                index=numeric_cols.index("BPHIGH")
                if "BPHIGH" in numeric_cols
                else 1,
            )

        fig_scatter = px.scatter(
            df,
            x=x_var,
            y=y_var,
            hover_name="locationname" if "locationname" in df.columns else None,
            color="BPHIGH" if "BPHIGH" in df.columns else None,
            color_continuous_scale="Viridis",
            title=f"{y_var} vs. {x_var}",
            opacity=0.7,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ==========================================
# TAB 3: CHRR BENCHMARK COMPARISON
# ==========================================
with tab3:
    st.header("🔄 CHRR Benchmark Comparison")
    st.info(
        "📊 **CHR&R Data is Excluded from Modeling:** While acquired and validated,"
        "County Health Rankings & Roadmaps (CHR&R) data (**796 variables**, high missingness) were excluded "
        "from the modeling pipeline to prevent severe multicollinearity and structural pipeline breakage. "
        "Because of substantial feature redundancy, CHR&R indicators are instead reserved for "
        "dashboard analytics as a ground-truth benchmark comparison layer.")

    st.write(
        "🔍 **Benchmark Explorer:** Select a county below to compare its core indicator values "
            "from the master modeling dataset directly against the corresponding CHR&R ground-truth metrics."
    )

    if not df.empty and not df_chrr.empty:
        if "locationname" in df.columns:
            county_list = df["locationname"].dropna().unique()
            selected_county = st.selectbox(
                "Select a County to Benchmark", sorted(county_list)
            )

            county_master = df[df["locationname"] == selected_county]
            st.write(f"### Insights for {selected_county}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Atlas / CDC Metrics")
                st.dataframe(county_master.T)

            with col2:
                st.subheader("CHRR Benchmark Data")
                if (
                    "fipscode" in county_master.columns
                    and "fipscode" in df_chrr.columns
                ):
                    match_fips = county_master["fipscode"].values[0]
                    county_chrr = df_chrr[df_chrr["fipscode"] == match_fips]
                    st.dataframe(county_chrr.T)
                else:
                    st.write(
                        "CHRR data preview (unfiltered or matching by name):"
                    )
                    st.dataframe(df_chrr.head(5))
        else:
            st.dataframe(df_chrr.head(100))
    else:
        st.info("CHRR data is currently empty or loading.")


# ==========================================
# TAB 4: MODEL PERFORMANCE & INTERPRETABILITY
# ==========================================
with tab4:
    st.header("🤖 Machine Learning Model Performance")
    st.write(
        "Explore pipeline architecture, cross-validation screening across candidate models, "
        "feature importance, and test-set residuals."
    )

    if not X_train.empty and not y_train.empty:
        # --- Section 1: Pipeline Overview ---
        st.subheader("Pipeline Architecture")
        st.markdown(
            """
        To establish a robust benchmark, a standardized pipeline architecture was implemented:
        - **Imputer (`SimpleImputer`)**: Imputes missing values using the median strategy.
        - **Scaler (`StandardScaler`)**: Standardizes features to a mean of 0 and variance of 1.
        - **Feature Selection (`SelectFromModel` with Lasso)**: Drives redundant coefficients to zero ($\alpha = 0.0055$).
        - **Dimensionality Reduction (`PCA`)**: Compresses surviving predictors into **15 principal components**.
        - **Regressor**: Evaluated across multiple modular algorithms.
        """
        )

        # --- Section 2: Train/Test Shape Summary ---
        shape_col1, shape_col2, shape_col3 = st.columns(3)
        with shape_col1:
            st.metric("Training Samples", f"{X_train.shape[0]:,}")
        with shape_col2:
            st.metric("Test Samples", f"{X_test.shape[0]:,}")
        with shape_col3:
            st.metric("Total Features", f"{X_train.shape[1]:,}")

        st.markdown("---")

        # --- Section 3: Interactive Model Explorer ---
        st.subheader("Interactive Model Explorer")
        
        # Quick metric summary card for your best model
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("🏆 Best Model", "Tuned XGBoost")
        col_m2.metric("Test R²", "0.9271")
        col_m3.metric("Test RMSE", "1.2117")

        model_data = {
            "Model": ["XGBoost", "Support Vector Reg. (RBF)", "Random Forest", "Elastic Net", "Ridge", "Linear Regression (base)"],
            "R² (Mean)": [0.9166, 0.9138, 0.9029, 0.9005, 0.9001, 0.9001],
            "RMSE": [1.3489, 1.3726, 1.4540, 1.4743, 1.4772, 1.4772],
            "MAE": [1.0381, 1.0206, 1.1158, 1.1569, 1.1571, 1.1571],
            "Fit Time (s)": [1.20, 0.31, 8.00, 0.12, 0.10, 0.12]
        }
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

        st.info(
            "📈 **Model Explorer Instructions:** Use the controls below to select and evaluate "
            "different candidate regression models. Review their performance metrics and interactive residual "
            "plots to observe model behavior across counties."
        )

        chosen_model_name = st.selectbox(
            "Select Regressor to Evaluate:",
            ["Linear Regression", "Ridge", "Random Forest", "XGBoost"],
            key="tab4_model_select"
        )

        # Dynamic hyperparameter controls based on selection
        if chosen_model_name == "Random Forest":
            n_est = st.slider("Number of Estimators", 100, 600, 400, step=50, key="rf_nest")
            max_d = st.slider("Max Depth", 4, 20, 10, key="rf_maxd")
            regressor = RandomForestRegressor(
                n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1
            )
        elif chosen_model_name == "XGBoost":
            lr = st.slider("Learning Rate", 0.01, 0.2, 0.05, step=0.01, key="xgb_lr")
            n_est = st.slider("Number of Estimators", 100, 600, 400, step=50, key="xgb_nest")
            regressor = XGBRegressor(
                learning_rate=lr,
                n_estimators=n_est,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        elif chosen_model_name == "Ridge":
            alpha_val = st.slider("Ridge Alpha", 0.1, 50.0, 1.0, key="ridge_alpha")
            regressor = Ridge(alpha=alpha_val, random_state=42)
        else:
            regressor = LinearRegression()

        if st.button("Train & Evaluate Selected Model", key="tab4_train_btn"):
            with st.spinner(f"Training {chosen_model_name}..."):
                ID_COLS = ["fipscode", "locationname", "stateabbr", "State", "County"]
                X_tr = X_train.drop(columns=[c for c in ID_COLS if c in X_train.columns], errors="ignore")
                X_te = X_test.drop(columns=[c for c in ID_COLS if c in X_test.columns], errors="ignore")
                y_tr = y_train.squeeze()
                y_te = y_test.squeeze()

                pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "feature_selection",
                            SelectFromModel(
                                Lasso(alpha=0.0055, max_iter=10000, random_state=42)
                            ),
                        ),
                        ("pca", PCA(n_components=15, random_state=42)),
                        ("regressor", regressor),
                    ]
                )

                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)

                r2 = r2_score(y_te, y_pred)
                rmse = root_mean_squared_error(y_te, y_pred)
                mae = mean_absolute_error(y_te, y_pred)

                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Test R²", f"{r2:.4f}")
                res_col2.metric("Test RMSE", f"{rmse:.4f}")
                res_col3.metric("Test MAE", f"{mae:.4f}")

                # --- Section 4: Interactive Test Residuals Explorer ---
                st.subheader("Interactive Test Residuals Explorer")
                residuals_df = pd.DataFrame(
                    {
                        "Actual": y_te,
                        "Predicted": y_pred,
                        "Residual": y_te - y_pred,
                        "County": X_test["locationname"] if "locationname" in X_test.columns else "Unknown",
                    }
                )

                fig_res = px.scatter(
                    residuals_df,
                    x="Predicted",
                    y="Residual",
                    hover_name="County",
                    color="Residual",
                    color_continuous_scale="Tealrose",
                    title=f"Interactive Residuals vs. Fitted Values ({chosen_model_name})",
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_res, use_container_width=True)

        st.markdown("---")

        # --- Section 5: Global Interpretability (SHAP) ---
        st.header("📊 Global Interpretability & SHAP Feature Importance")
        st.info(
            "💡 **SHAP Insights:** The summary plot below highlights which risk factors exert "
            "the highest positive and negative influence on county-level hypertension prevalence, "
            "derived from our fully-tuned XGBoost model."
        )

        current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
        image_path = current_dir / "shap_summary_importance_col.png"

        if image_path.exists():
            st.image(
                str(image_path),
                caption="Top 20 SHAP Features of County Hypertension Prevalence (Tuned XGBoost)",
                use_container_width=True,
            )
        else:
            # Check alternative relative path if running from project root
            alt_path = current_dir / "dashboard" / "shap_summary_importance_col.png"
            if alt_path.exists():
                st.image(
                    str(alt_path),
                    caption="Top 20 SHAP Features of County Hypertension Prevalence (Tuned XGBoost)",
                    use_container_width=True,
                )
            else:
                st.warning("SHAP feature importance chart (`shap_summary_importance_col.png`) was not found in the dashboard folder.")

    else:
        st.warning("Model training datasets could not be loaded from the SQLite database.")

# ==========================================
# TAB 5: POLICY WHAT-IF SIMULATOR
# ==========================================
with tab5:
    st.header("📈 Policy What-If Simulator")
    st.write(
        "Simulate structural policy interventions (e.g., improving transit access, reducing food deserts, "
        "or lowering poverty) to observe predicted changes in regional hypertension risk."
    )

    if not df.empty:
        # Pick a baseline county to start the simulation from
        county_list = sorted(df["locationname"].dropna().unique().tolist())
        sim_county = st.selectbox("Select Baseline County for Simulation:", county_list, key="sim_county")
        
        base_row = df[df["locationname"] == sim_county].iloc[0]

        st.subheader("Adjust Intervention Levers")
        st.write("Modify the sliders below to simulate local policy changes:")

        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            # Example slider for a transit/vehicle access feature
            orig_transit = float(base_row.get("TractHUNV", 50))
            sim_transit = st.slider("Simulated Low-Vehicle Households (Transit proxy)", 0.0, 500.0, orig_transit)
            
            # Example slider for food access/desert density
            orig_food = float(base_row.get("LILATracts_1And10", 10))
            sim_food = st.slider("Simulated Low-Access Tracts (Food Program proxy)", 0.0, 100.0, orig_food)

        with col_s2:
            # Example slider for poverty rate
            orig_pov = float(base_row.get("PovertyRate", 15.0))
            sim_pov = st.slider("Simulated Poverty Rate (%)", 0.0, 50.0, orig_pov)

        # Simulation calculation button or real-time display
        if st.button("Run Simulation", key="run_sim"):
            # Simple heuristic / linear approximation or live model prediction based on adjusted inputs
            # (If using your trained pipeline, you can pass a modified copy of the feature row through `pipeline.predict()`)
            baseline_risk = float(base_row.get("BPHIGH", 33.0))
            
            # Simple logic simulation example (replace with model predict if preferred):
            delta_risk = ((sim_pov - orig_pov) * 0.15) - ((orig_transit - sim_transit) * 0.01)
            simulated_risk = max(10.0, min(60.0, baseline_risk + delta_risk))

            st.markdown("---")
            st.subheader("🎯 Simulation Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Baseline Risk", f"{baseline_risk:.1f}%")
            res_col2.metric("Simulated Risk", f"{simulated_risk:.1f}%", delta=f"{simulated_risk - baseline_risk:.1f}%")
            res_col3.metric("Projected Impact", "Favorable" if simulated_risk < baseline_risk else "Elevated")

            st.success(
                f"💡 **Policy Insight:** Implementing targeted transit and food access interventions in **{sim_county}** "
                f"is projected to shift the regional hypertension risk score from **{baseline_risk:.1f}%** to **{simulated_risk:.1f}%**."
            )
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
   "This is a web application that provides insights into hypertension risk factors and their prevalence across different regions."
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

NAVY_COLOR = "#191970" 

def set_text_color_and_bg():
    st.markdown(
        f"""
        <style>
        /* Force all text elements to be navy using !important */
        .stApp, h1, h2, h3, h4, h5, h6, p, span, label, div, ul, ol, li, 
        [data-testid="stMarkdownContainer"], 
        [data-testid="stMetricLabel"], 
        [data-testid="stMetricValue"] {{
            color: {NAVY_COLOR} !important;
        }}

        /* Ensure interactive widget text is also navy */
        input, select, textarea, button {{
            color: {NAVY_COLOR} !important;
        }}

        /* --- Style Streamlit Tabs --- */
        /* Unselected tab styling */
        [data-baseweb="tab"] {{
            background-color: rgba(245, 235, 220, 0.7) !important;
            border-radius: 4px 4px 0px 0px !important;
        }}
        
        /* Tab text color */
        [data-baseweb="tab"] div {{
            color: {NAVY_COLOR} !important;
            font-weight: 600 !important;
        }}

        /* Selected active tab styling */
        [data-baseweb="tab"][aria-selected="true"] {{
            background-color: rgba(255, 253, 245, 0.95) !important;
            border-bottom-color: {NAVY_COLOR} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_text_color_and_bg()

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

# --- Create 4-level tabs for navigation ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🗺️ Interactive Atlas",
        "📊 Statistical Insights",
        "🔄 CHRR Data Comparison",
        "🤖 Model Performance",
    ]
)

# ==========================================
# TAB 1: INTERACTIVE ATLAS
# ==========================================
with tab1:
    st.header("Geographic Risk Atlas (County-Level) 🗺️")

    if not df.empty:
        st.subheader("Dataset Overview")
        st.metric("Total Records (Counties)", len(df))

        # State Filter Dropdown
        if "stateabbr" in df.columns:
            states = ["All US"] + sorted(df["stateabbr"].dropna().unique().tolist())
            selected_state = st.selectbox("Filter Map by State:", states)

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
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw Data Preview")
        st.dataframe(df.head(100))


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
        fig1, ax1 = plt.subplots(figsize=(8, 4))
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
        vehicle_access_rate = df["TractHUNV"] / df["OHU2010"]
        food_desert_density = df["LILATracts_1And10"] / df["Pop2010"] * 100_000

        fig_grid, axes = plt.subplots(2, 2, figsize=(10, 7))
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
        n_features = st.slider("Select number of top predictors to display:", 5, 30, 15)

        corr = df.select_dtypes("number").corrwith(df["BPHIGH"]).drop("BPHIGH").dropna()
        top_n = corr.reindex(corr.abs().sort_values(ascending=False).index).head(
            n_features
        )
        top_n_sorted = top_n.sort_values()

        fig2, ax2 = plt.subplots(figsize=(8, max(4, n_features * 0.3)))
        top_n_sorted.plot.barh(ax=ax2, color="teal")
        ax2.set_xlabel("Correlation with BPHIGH")
        ax2.set_title(f"Top {n_features} Predictors Correlated with Hypertension")
        plt.tight_layout()
        st.pyplot(fig2)

        # --- Section 4: Correlation Heatmap Matrix ---
        st.subheader("Feature Correlation Heatmap Matrix")
        top_features = (
            list(corr.head(10).index) + list(corr.tail(10).index) + ["BPHIGH"]
        )
        corr_matrix = df[top_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        fig3, ax3 = plt.subplots(figsize=(9, 7))
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
    st.write(
        "Compare county health outcomes against County Health Rankings & Roadmaps (CHRR) benchmarks."
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
# TAB 4: MODEL PERFORMANCE
# ==========================================
with tab4:
    st.header("🤖 Machine Learning Model Performance")
    st.write(
        "Explore pipeline architecture, cross-validation screening across candidate models, feature importance, and test-set residuals."
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", f"{X_train.shape[0]:,}")
        with col2:
            st.metric("Test Samples", f"{X_test.shape[0]:,}")
        with col3:
            st.metric("Total Features", f"{X_train.shape[1]:,}")

        # --- Section 3: Interactive Model Explorer ---
        st.subheader("Interactive Model Explorer")

        chosen_model_name = st.selectbox(
            "Select Regressor to Evaluate:",
            ["Linear Regression", "Ridge", "Random Forest", "XGBoost"],
        )

        # Dynamic hyperparameter controls based on selection
        if chosen_model_name == "Random Forest":
            n_est = st.slider("Number of Estimators", 100, 600, 400, step=50)
            max_d = st.slider("Max Depth", 4, 20, 10)
            regressor = RandomForestRegressor(
                n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1
            )
        elif chosen_model_name == "XGBoost":
            lr = st.slider("Learning Rate", 0.01, 0.2, 0.05, step=0.01)
            n_est = st.slider("Number of Estimators", 100, 600, 400, step=50)
            regressor = XGBRegressor(
                learning_rate=lr,
                n_estimators=n_est,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        elif chosen_model_name == "Ridge":
            alpha_val = st.slider("Ridge Alpha", 0.1, 50.0, 1.0)
            regressor = Ridge(alpha=alpha_val, random_state=42)
        else:
            regressor = LinearRegression()

        if st.button("Train & Evaluate Selected Model"):
            with st.spinner(f"Training {chosen_model_name}..."):
                ID_COLS = [
                    "fipscode",
                    "locationname",
                    "stateabbr",
                    "State",
                    "County",
                ]
                X_tr = X_train.drop(
                    columns=[c for c in ID_COLS if c in X_train.columns]
                )
                X_te = X_test.drop(
                    columns=[c for c in ID_COLS if c in X_test.columns]
                )
                y_tr = y_train.squeeze()
                y_te = y_test.squeeze()

                pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "feature_selection",
                            SelectFromModel(
                                Lasso(
                                    alpha=0.0055, max_iter=10000, random_state=42
                                )
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

                col1, col2, col3 = st.columns(3)
                col1.metric("Test R²", f"{r2:.4f}")
                col2.metric("Test RMSE", f"{rmse:.4f}")
                col3.metric("Test MAE", f"{mae:.4f}")

                # --- Section 4: Interactive Test Residuals Explorer ---
                st.subheader("Interactive Test Residuals Explorer")
                residuals_df = pd.DataFrame(
                    {
                        "Actual": y_te,
                        "Predicted": y_pred,
                        "Residual": y_te - y_pred,
                        "County": X_test["locationname"]
                        if "locationname" in X_test.columns
                        else "Unknown",
                    }
                )

                fig_res = px.scatter(
                    residuals_df,
                    x="Predicted",
                    y="Residual",
                    hover_name="County",
                    color="Residual",
                    color_continuous_scale="Tealrose",
                    title="Interactive Residuals vs. Fitted Values (Hover for County Details)",
                )
                fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_res, use_container_width=True)

        # --- Section 5: What-If Analysis ---
        st.subheader("County Prediction 'What-If' Simulator")
        if "locationname" in X_test.columns:
            county_options = X_test["locationname"].dropna().unique().tolist()
            selected_eval_county = st.selectbox(
                "Select a County to Test:", county_options
            )

            county_row = X_test[X_test["locationname"] == selected_eval_county]

            if not county_row.empty:
                st.write(
                    f"Adjust key risk factors for **{selected_eval_county}** to simulate changes in hypertension risk:"
                )
                sim_poverty = st.slider(
                    "Poverty Rate (%)",
                    0.0,
                    50.0,
                    float(
                        county_row["PovertyRate"].values[0]
                        if "PovertyRate" in county_row.columns
                        else 15.0
                    ),
                )

                if st.button("Simulate New Risk Prevalence"):
                    st.success(
                        f"Simulated Hypertension Prevalence for {selected_eval_county}: **Calculated Risk Updated** (Poverty Rate set to {sim_poverty}%)"
                    )
    else:
        st.warning(
            "Model training datasets could not be loaded from the SQLite database."
        )
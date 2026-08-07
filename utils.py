from paths import DATA_RAW, DATA_PROCESSED
import hashlib
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

def get_file_hash(file_path):
    """Generates an MD5 hash of your data file to ensure integrity."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_missingness_summary(df):
    """
    Analyzes a dataframe and returns a summary table of missingness tiers.
    """
    missing_pct = df.isnull().mean() * 100
    
    summary = pd.DataFrame({
        "Missing Threshold": ["0%", "<5%", "5–20%", "20–40%", "40–60%", ">60%"],
        "Count": [
            (missing_pct == 0).sum(),
            ((missing_pct > 0) & (missing_pct < 5)).sum(),
            ((missing_pct >= 5) & (missing_pct < 20)).sum(),
            ((missing_pct >= 20) & (missing_pct < 40)).sum(),
            ((missing_pct >= 40) & (missing_pct < 60)).sum(),
            (missing_pct >= 60).sum()
        ]
    })
    return summary

# Publication color registry shared with all features .
VARIABLE_COLORS = {
    "BPHIGH": "#0072B2", "predicted_BPHIGH": "#E69F00",
    "PovertyRate": "#CC79A7",
    "residual": "#CC79A7", "reference": "#333333",
    "individual_explained_variance": "#56B4E9",
    "cumulative_explained_variance": "#3B528B",
    "DIABETES": "#1F77B4", "MOBILITY": "#AEC7E8",
    "FOODINSECU": "#FF7F0E", "SELFCARE": "#FFBB78",
    "HOUSINSECU": "#2CA02C", "LACKTRPT": "#98DF8A",
    "LPA": "#D62728", "GHLTH": "#FF9896",
    "COPD": "#9467BD", "SLEEP": "#C5B0D5",
    "COGNITION": "#8C564B", "FOODSTAMP": "#C49C94",
    "VISION": "#E377C2", "TEETHLOST": "#F7B6D2",
    "OBESITY": "#7F7F7F", "LONELINESS": "#C7C7C7",
    "CSMOKING": "#BCBD22", "EMOTIONSPT": "#17BECF",
    "CASTHMA": "#393B79", "lapophalfshare": "#637939",
    "LATracts1": "#8C6D31", "laaianhalfshare": "#843C39",
    "lahunv1share": "#7B4173", "LATracts10": "#5254A3",
    "TractNHOPI": "#8CA252", "ACCESS2": "#BD9E39",
    "lablackhalfshare": "#DBDB8D", "lahisphalfshare": "#9C9EDE",
    "LILATracts_halfAnd10": "#5254A3",
}
MODEL_COLORS = {
    "Linear Regression (baseline)": "#4E79A7", "Ridge": "#F28E2B",
    "ElasticNet": "#E15759", "SVR (RBF)": "#76B7B2",
    "Random Forest": "#59A14F", "XGBoost": "#EDC948",
}
FALLBACK_COLORS = sum(
    (list(plt.get_cmap(name).colors) for name in ("tab20", "tab20b", "tab20c")),
    [],
)

def variable_color(name):
    """Return a persistent color and safely register unseen predictors."""
    if name not in VARIABLE_COLORS:
        digest = hashlib.sha256(str(name).encode("utf-8")).digest()
        VARIABLE_COLORS[name] = FALLBACK_COLORS[int.from_bytes(digest[:4], "big") % len(FALLBACK_COLORS)]
    return VARIABLE_COLORS[name]

def fig_to_base64(fig):
    """Utility to convert a Matplotlib figure to a base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"
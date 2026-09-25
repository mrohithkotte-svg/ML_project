import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATHS & CONFIG
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "placement_predict_50k Dataset (3) 1(in).csv"
)

CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

FEATURES = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]

VALID_LINKAGE_METHODS = ["single", "complete", "average", "ward"]


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

def load_hierarchical_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    data = df[FEATURES].copy()

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna()
    return df, data


def prepare_hierarchical_data(sample_size=150):
    original_df, data = load_hierarchical_data()

    # Sample data for clear dendrogram visualization and performance
    if len(data) > sample_size:
        data_sample = data.head(sample_size).copy()
    else:
        data_sample = data.copy()

    scaler = StandardScaler()
    scaled_sample = scaler.fit_transform(data_sample)

    return original_df, data_sample, scaled_sample


# =========================================================
# RUN HIERARCHICAL CLUSTERING
# =========================================================

def run_hierarchical_clustering(linkage_method="complete"):
    linkage_method = str(linkage_method).lower().strip()

    if linkage_method not in VALID_LINKAGE_METHODS:
        raise ValueError(
            f"Invalid linkage method '{linkage_method}'. Must be one of: {', '.join(VALID_LINKAGE_METHODS)}"
        )

    original_df, data_sample, scaled_sample = prepare_hierarchical_data(sample_size=150)

    # Perform hierarchical clustering linkage
    # Scipy linkage method handles 'ward', 'single', 'complete', 'average'
    # Metric is Euclidean distance
    Z = linkage(scaled_sample, method=linkage_method, metric="euclidean")

    # Generate dendrogram chart
    fig, ax = plt.subplots(figsize=(10, 6))

    dendrogram(
        Z,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=None
    )

    ax.set_title(f"Hierarchical Clustering Dendrogram ({linkage_method.capitalize()} Linkage)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sample Index", fontsize=11, labelpad=10)
    ax.set_ylabel("Euclidean Distance", fontsize=11, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    chart_filename = "hierarchical_dendrogram.png"
    chart_path = os.path.join(CHARTS_DIR, chart_filename)

    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result_df = data_sample.copy()

    return {
        "method": linkage_method.capitalize(),
        "distance_metric": "Euclidean",
        "n_samples": len(data_sample),
        "total_rows": len(original_df),
        "chart": chart_filename,
        "data": result_df.head(20).to_dict(orient="records")
    }

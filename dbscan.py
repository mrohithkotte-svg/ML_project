import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


# =========================================================
# LOAD AND PREPARE DATA
# =========================================================

def load_dbscan_data():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    data = df[FEATURES].copy()

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna()
    return df, data


def prepare_dbscan_data():
    original_df, data = load_dbscan_data()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return original_df, data, scaled_data


# =========================================================
# RUN DBSCAN
# =========================================================

def run_dbscan(eps=0.5, min_samples=5):
    # Validation
    try:
        eps = float(eps)
    except (ValueError, TypeError):
        raise ValueError("Epsilon (eps) must be a valid positive number.")

    if eps <= 0:
        raise ValueError("Epsilon (eps) must be a positive number greater than 0.")

    try:
        min_samples = int(min_samples)
    except (ValueError, TypeError):
        raise ValueError("Minimum Samples (min_samples) must be a valid positive integer.")

    if min_samples < 1:
        raise ValueError("Minimum Samples (min_samples) must be an integer of at least 1.")

    original_df, data, scaled_data = prepare_dbscan_data()

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled_data)

    # Number of clusters (ignoring noise point label -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))
    n_samples = len(scaled_data)

    result_df = data.copy()
    result_df["Cluster"] = labels

    # Cluster distribution map
    cluster_counts = result_df["Cluster"].value_counts().sort_index().to_dict()
    # Format cluster labels in dictionary for cleaner UI presentation
    formatted_cluster_counts = {}
    for cluster_id, count in cluster_counts.items():
        if cluster_id == -1:
            formatted_cluster_counts["Noise Points (-1)"] = int(count)
        else:
            formatted_cluster_counts[f"Cluster {cluster_id}"] = int(count)

    # Generate Cluster Visualization
    # To plot 4D feature space in 2D, use 2D PCA or 2 main features (CGPA vs CodingTestScore)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Perform 2D PCA projection for clean visual clustering representation
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(scaled_data)

    # Plot noise points and clusters
    unique_sorted = sorted(list(unique_labels))
    colors = plt.get_cmap("Spectral")

    for idx, cluster_id in enumerate(unique_sorted):
        mask = (labels == cluster_id)
        if cluster_id == -1:
            # Noise points marked with black cross
            ax.scatter(
                pca_data[mask, 0],
                pca_data[mask, 1],
                c="black",
                marker="x",
                s=20,
                alpha=0.6,
                label="Noise (-1)"
            )
        else:
            ax.scatter(
                pca_data[mask, 0],
                pca_data[mask, 1],
                color=colors(idx),
                s=15,
                alpha=0.7,
                label=f"Cluster {cluster_id}"
            )

    ax.set_title(f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})", fontsize=13, fontweight="bold")
    ax.set_xlabel("PCA Component 1", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    if len(unique_labels) <= 12:
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()

    chart_filename = "dbscan_chart.png"
    chart_path = os.path.join(CHARTS_DIR, chart_filename)

    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_samples": n_samples,
        "cluster_counts": formatted_cluster_counts,
        "chart": chart_filename,
        "data": result_df.head(20).to_dict(orient="records")
    }

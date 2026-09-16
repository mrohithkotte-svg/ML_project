import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "placement_predict_50k Dataset (3) 1(in).csv"
)

CHARTS_DIR = os.path.join(BASE_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)


# =========================================================
# FEATURES USED FOR K-MEANS
# =========================================================

FEATURES = [
    "CGPA",
    "AptitudeTestScore",
    "CodingTestScore",
    "MockInterviewScore"
]


# =========================================================
# LOAD DATA
# =========================================================

def load_kmeans_data():

    df = pd.read_csv(DATASET_PATH)

    # Keep only required features
    data = df[FEATURES].copy()

    # Convert to numeric
    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove missing values
    data = data.dropna()

    return df, data


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data():

    original_df, data = load_kmeans_data()

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data)

    return original_df, data, scaled_data


# =========================================================
# MANUAL K
# =========================================================

def run_manual_k(k):

    original_df, data, scaled_data = prepare_data()

    k = int(k)

    if k < 2:
        raise ValueError("K must be at least 2.")

    if k >= len(data):
        raise ValueError("K must be smaller than the number of data points.")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled_data)

    score = silhouette_score(
        scaled_data,
        clusters
    )

    result_df = data.copy()

    result_df["Cluster"] = clusters

    return {
        "method": "Manual K",
        "k": k,
        "silhouette_score": round(score, 4),
        "inertia": round(model.inertia_, 4),
        "cluster_counts": result_df["Cluster"].value_counts().sort_index().to_dict(),
        "data": result_df.head(20).to_dict(orient="records")
    }


# =========================================================
# ELBOW METHOD
# =========================================================

def run_elbow_method():

    original_df, data, scaled_data = prepare_data()

    max_k = min(10, len(data) - 1)

    k_values = range(2, max_k + 1)

    inertias = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(scaled_data)

        inertias.append(model.inertia_)

    # Simple automatic elbow selection
    # Uses the point with maximum distance from
    # the line joining first and last points.

    x = list(k_values)
    y = inertias

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    distances = []

    for xi, yi in zip(x, y):

        numerator = abs(
            (y2 - y1) * xi
            - (x2 - x1) * yi
            + x2 * y1
            - y2 * x1
        )

        denominator = (
            (y2 - y1) ** 2
            + (x2 - x1) ** 2
        ) ** 0.5

        distance = numerator / denominator

        distances.append(distance)

    best_index = distances.index(max(distances))

    selected_k = x[best_index]

    # Plot elbow graph

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        y,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")

    plt.title("K-Means Elbow Method")

    plt.xticks(x)

    plt.grid(True)

    elbow_chart = os.path.join(
        CHARTS_DIR,
        "kmeans_elbow.png"
    )

    plt.savefig(
        elbow_chart,
        bbox_inches="tight"
    )

    plt.close()

    # Run final K-Means using selected K

    model = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled_data)

    result_df = data.copy()

    result_df["Cluster"] = clusters

    score = silhouette_score(
        scaled_data,
        clusters
    )

    return {
        "method": "Elbow Method",
        "k": selected_k,
        "silhouette_score": round(score, 4),
        "inertia": round(model.inertia_, 4),
        "cluster_counts": result_df["Cluster"].value_counts().sort_index().to_dict(),
        "data": result_df.head(20).to_dict(orient="records"),
        "chart": "kmeans_elbow.png"
    }


# =========================================================
# SILHOUETTE METHOD
# =========================================================

def run_silhouette_method():

    original_df, data, scaled_data = prepare_data()

    max_k = min(10, len(data) - 1)

    k_values = range(2, max_k + 1)

    scores = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        clusters = model.fit_predict(scaled_data)

        score = silhouette_score(
            scaled_data,
            clusters
        )

        scores.append(score)

    # Select K with highest silhouette score

    best_index = scores.index(max(scores))

    selected_k = list(k_values)[best_index]

    best_score = scores[best_index]

    # Plot silhouette scores

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        scores,
        marker="o"
    )

    plt.xlabel("Number of Clusters (K)")

    plt.ylabel("Silhouette Score")

    plt.title("K-Means Silhouette Method")

    plt.xticks(list(k_values))

    plt.grid(True)

    silhouette_chart = os.path.join(
        CHARTS_DIR,
        "kmeans_silhouette.png"
    )

    plt.savefig(
        silhouette_chart,
        bbox_inches="tight"
    )

    plt.close()

    # Final model

    model = KMeans(
        n_clusters=selected_k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(scaled_data)

    result_df = data.copy()

    result_df["Cluster"] = clusters

    return {
        "method": "Silhouette Method",
        "k": selected_k,
        "silhouette_score": round(best_score, 4),
        "inertia": round(model.inertia_, 4),
        "cluster_counts": result_df["Cluster"].value_counts().sort_index().to_dict(),
        "data": result_df.head(20).to_dict(orient="records"),
        "chart": "kmeans_silhouette.png"
    }
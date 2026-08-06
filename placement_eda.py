import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

# Save charts inside Flask's static folder
CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static")


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def _save(filename):
    plt.tight_layout()
    plt.savefig(_chart_path(filename), dpi=300, bbox_inches="tight")
    plt.close()


def run_eda():
    data = load_data()

    charts = []

    # Seaborn settings
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # Pandas settings
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("=" * 80)
    print("DATA LOADED")
    print("=" * 80)
    print(data.head())

    # -----------------------------
    # Missing Values
    # -----------------------------
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    missing_df = missing_df[missing_df["missing_count"] > 0]

    print("\nMissing Values")
    print(missing_df)

    if not missing_df.empty:

        plt.figure(figsize=(10, 5))
        sns.barplot(
            x=missing_df.index,
            y=missing_df["missing_pct"]
        )
        plt.xticks(rotation=45)
        plt.ylabel("Missing %")
        plt.title("Missing Values")
        _save("missing_values.png")
        charts.append("missing_values.png")

        plt.figure(figsize=(12, 6))
        sns.heatmap(data.isnull(), cbar=False)
        plt.title("Missing Value Heatmap")
        _save("missing_heatmap.png")
        charts.append("missing_heatmap.png")

    # -----------------------------
    # Duplicate Rows
    # -----------------------------
    duplicates = data.duplicated().sum()

    print("\nDuplicate Rows:", duplicates)

    # -----------------------------
    # Placement Status
    # -----------------------------
    print("\nPlacement Status")
    print(data["PlacementStatus"].value_counts())
    target_counts = data["PlacementStatus"].value_counts().to_dict()

    plt.figure(figsize=(6, 5))
    sns.countplot(x="PlacementStatus", data=data)
    plt.title("Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    _save("placement_status.png")
    charts.append("placement_status.png")

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": duplicates,
        "missing": missing_df["missing_count"].to_dict(),
        "target_counts": target_counts,
        "charts": charts
    }


if __name__ == "__main__":
    result = run_eda()
    print(result)
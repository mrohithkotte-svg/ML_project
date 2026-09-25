from flask import Flask, render_template, request

from load_data import get_data_summary
from placement_eda import run_eda
from preprocessing import run_preprocessing
from linear_regression import run_linear_regression
from Logistic_Regression import run_logistic_regression
from decision_tree import run_decision_tree
from random_forest import run_random_forest
from bagging import run_bagging
from boosting import run_boosting

from kmeans import (
    run_manual_k,
    run_elbow_method,
    run_silhouette_method
)
from hierarchical import run_hierarchical_clustering as run_hierarchical
from dbscan import run_dbscan



# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        active="none",
        summary=None,
        error=None
    )


# =========================================================
# DATA LOADING
# =========================================================

@app.route("/data-loading")
def data_loading():

    try:

        summary = get_data_summary()

        return render_template(
            "index.html",
            active="data-loading",
            summary=summary,
            error=None
        )

    except Exception as e:

        return render_template(
            "index.html",
            active="data-loading",
            summary=None,
            error=str(e)
        )


# =========================================================
# EDA
# =========================================================

@app.route("/eda")
def eda():

    try:

        result = run_eda()

        return render_template(
            "eda.html",
            active="eda",
            eda=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "eda.html",
            active="eda",
            eda=None,
            error=str(e)
        )


# =========================================================
# PREPROCESSING
# =========================================================

@app.route("/preprocessing")
def preprocessing():

    try:

        result = run_preprocessing()

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            preprocessing=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            preprocessing=None,
            error=str(e)
        )


# =========================================================
# LINEAR REGRESSION
# =========================================================

@app.route("/linear-regression")
def linear_regression():

    try:

        result = run_linear_regression()

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            regression=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "linear_regression.html",
            active="linear-regression",
            regression=None,
            error=str(e)
        )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

@app.route("/logistic-regression")
def logistic_regression():

    try:

        result = run_logistic_regression()

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            logistic=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "logistic_regression.html",
            active="logistic-regression",
            logistic=None,
            error=str(e)
        )


# =========================================================
# DECISION TREE
# =========================================================

@app.route("/decision-tree")
def decision_tree():

    try:

        result = run_decision_tree()

        return render_template(
            "decision_tree.html",
            active="decision-tree",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "decision_tree.html",
            active="decision-tree",
            result=None,
            error=str(e)
        )


# =========================================================
# RANDOM FOREST
# =========================================================

@app.route("/random-forest")
def random_forest():

    try:

        result = run_random_forest()

        return render_template(
            "random_forest.html",
            active="random-forest",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "random_forest.html",
            active="random-forest",
            result=None,
            error=str(e)
        )


# =========================================================
# BAGGING
# =========================================================

@app.route("/bagging")
def bagging():

    try:

        result = run_bagging()

        return render_template(
            "bagging.html",
            active="bagging",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "bagging.html",
            active="bagging",
            result=None,
            error=str(e)
        )


# =========================================================
# BOOSTING
# =========================================================

@app.route("/boosting")
def boosting():

    try:

        result = run_boosting()

        return render_template(
            "boosting.html",
            active="boosting",
            result=result,
            error=None
        )

    except Exception as e:

        return render_template(
            "boosting.html",
            active="boosting",
            result=None,
            error=str(e)
        )


# =========================================================
# CLUSTERING MODULES
# =========================================================

@app.route("/kmeans", methods=["GET", "POST"])
@app.route("/clustering", methods=["GET", "POST"])
@app.route("/clustering/kmeans", methods=["GET", "POST"])
def clustering_kmeans():

    result = None
    error = None

    method = "manual"
    manual_k = 3

    if request.method == "POST":

        method = request.form.get(
            "method",
            "manual"
        )

        try:

            # ---------------------------------------------
            # MANUAL K
            # ---------------------------------------------

            if method == "manual":

                manual_k = int(
                    request.form.get(
                        "k",
                        3
                    )
                )

                result = run_manual_k(
                    manual_k
                )

            # ---------------------------------------------
            # ELBOW METHOD
            # ---------------------------------------------

            elif method == "elbow":

                result = run_elbow_method()

            # ---------------------------------------------
            # SILHOUETTE METHOD
            # ---------------------------------------------

            elif method == "silhouette":

                result = run_silhouette_method()

        except Exception as e:

            error = str(e)

    return render_template(
        "kmeans.html",
        active="kmeans",
        result=result,
        error=error,
        method=method,
        manual_k=manual_k
    )


# Legacy endpoint alias for url_for('kmeans')
@app.route("/kmeans_legacy", methods=["GET", "POST"], endpoint="kmeans")
def kmeans():
    return clustering_kmeans()


@app.route("/clustering/hierarchical", methods=["GET", "POST"])
def clustering_hierarchical():

    result = None
    error = None
    linkage = "complete"

    if request.method == "POST":

        linkage = request.form.get(
            "linkage",
            "complete"
        )

        try:

            result = run_hierarchical(
                linkage_method=linkage
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "hierarchical.html",
        active="hierarchical",
        result=result,
        error=error,
        linkage=linkage
    )


@app.route("/clustering/dbscan", methods=["GET", "POST"])
def clustering_dbscan():

    result = None
    error = None
    eps = 0.5
    min_samples = 5

    if request.method == "POST":

        try:

            eps = float(
                request.form.get(
                    "eps",
                    0.5
                )
            )

            min_samples = int(
                request.form.get(
                    "min_samples",
                    5
                )
            )

            result = run_dbscan(
                eps=eps,
                min_samples=min_samples
            )

        except Exception as e:

            error = str(e)

    return render_template(
        "dbscan.html",
        active="dbscan",
        result=result,
        error=error,
        eps=eps,
        min_samples=min_samples
    )



# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
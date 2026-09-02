from flask import Flask, render_template

from load_data import get_data_summary
from placement_eda import run_eda
from preprocessing import run_preprocessing
from linear_regression import run_linear_regression


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
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
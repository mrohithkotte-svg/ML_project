from flask import Flask, render_template

from load_data import get_data_summary
from placement_eda import run_eda
from preprocessing import preprocess_data


app = Flask(__name__)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html",
        active="none"
    )


# ============================================================
# DATA LOADING
# ============================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:
        summary = get_data_summary()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ============================================================
# EDA
# ============================================================

@app.route("/eda")
def eda():

    error = None
    eda_output = None

    try:
        eda_output = run_eda()

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error
    )


# ============================================================
# PREPROCESSING
# ============================================================

@app.route("/preprocessing")
def preprocessing():

    error = None
    preprocessing_output = None

    try:

        # Run preprocessing
        (
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
            preprocessor
        ) = preprocess_data()

        # Create information to display in browser
        preprocessing_output = {

            "original_rows": 50000,
            "original_columns": 32,

            "train_rows": X_train_processed.shape[0],
            "test_rows": X_test_processed.shape[0],

            "train_features": X_train_processed.shape[1],
            "test_features": X_test_processed.shape[1],

            "y_train_size": y_train.shape[0],
            "y_test_size": y_test.shape[0],

            "duplicate_before": 0,
            "duplicate_after": 0,

            "missing_after": 0,

            "numerical_features": 20,
            "categorical_features": 7,

            "status": "Preprocessing completed successfully"
        }

    except FileNotFoundError as e:
        error = str(e)

    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="preprocessing",
        preprocessing=preprocessing_output,
        error=error
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
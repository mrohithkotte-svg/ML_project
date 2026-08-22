import pandas as pd
import numpy as np

from load_data import load_data

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data():

    print("\n========== PREPROCESSING STARTED ==========")

    # ==================================================
    # 1. LOAD DATA
    # ==================================================

    data = load_data()

    print("\nOriginal dataset shape:")
    print(data.shape)

    print("\nOriginal columns:")
    print(data.columns.tolist())

    # ==================================================
    # 2. REMOVE DUPLICATE ROWS
    # ==================================================

    print("\n========== DUPLICATE HANDLING ==========")

    duplicate_count = int(data.duplicated().sum())

    print("Duplicate rows before removal:", duplicate_count)

    if duplicate_count > 0:
        data = data.drop_duplicates()

    print("Duplicate rows after removal:", data.duplicated().sum())

    # ==================================================
    # 3. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
    # ==================================================

    numeric_columns = data.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # ==================================================
    # 4. COLUMNS NOT USED AS ML FEATURES
    # ==================================================

    exclude_columns = [
        "StudentID",
        "PlacementStatus",
        "IsAnomaly",
        "Salary Package",
        "CGPA_Tier"
    ]

    # Remove excluded columns from numerical list
    numeric_columns = [
        col for col in numeric_columns
        if col not in exclude_columns
    ]

    # Remove excluded columns from categorical list
    categorical_columns = [
        col for col in categorical_columns
        if col not in exclude_columns
    ]

    print("\n========== FEATURES USED FOR ML ==========")

    print("\nNumerical columns:")
    for col in numeric_columns:
        print(" -", col)

    print("\nCategorical columns:")
    for col in categorical_columns:
        print(" -", col)

    # ==================================================
    # 5. HANDLE MISSING NUMERICAL VALUES
    # ==================================================

    print("\n========== NUMERICAL MISSING VALUES ==========")

    for col in numeric_columns:

        missing_count = int(data[col].isnull().sum())

        if missing_count > 0:

            median_value = data[col].median()

            print(
                f"{col}: {missing_count} missing values "
                f"-> filling with median {median_value}"
            )

            data[col] = data[col].fillna(median_value)

    # ==================================================
    # 6. HANDLE MISSING CATEGORICAL VALUES
    # ==================================================

    print("\n========== CATEGORICAL MISSING VALUES ==========")

    for col in categorical_columns:

        missing_count = int(data[col].isnull().sum())

        if missing_count > 0:

            mode_value = data[col].mode()[0]

            print(
                f"{col}: {missing_count} missing values "
                f"-> filling with mode '{mode_value}'"
            )

            data[col] = data[col].fillna(mode_value)

    # ==================================================
    # 7. CHECK REMAINING MISSING VALUES
    # ==================================================

    print("\n========== MISSING VALUES AFTER HANDLING ==========")

    remaining_missing = data.isnull().sum()

    remaining_missing = remaining_missing[
        remaining_missing > 0
    ]

    if remaining_missing.empty:
        print("No missing values remaining.")
    else:
        print(remaining_missing)

    # ==================================================
    # 8. SEPARATE FEATURES AND TARGET
    # ==================================================

    print("\n========== TARGET SEPARATION ==========")

    target = "PlacementStatus"

    if target not in data.columns:
        raise ValueError(
            f"Target column '{target}' was not found."
        )

    X = data.drop(
        columns=[
            "PlacementStatus",
            "StudentID",
            "IsAnomaly",
            "Salary Package",
            "CGPA_Tier"
        ]
    )

    y = data[target]

    print("Target column:", target)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    # ==================================================
    # 9. TRAIN / TEST SPLIT
    # ==================================================

    print("\n========== TRAIN / TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ==================================================
    # 10. PREPROCESSING PIPELINE
    # ==================================================

    print("\n========== ENCODING AND SCALING ==========")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_columns
            ),

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ]
    )

    # ==================================================
    # 11. FIT PREPROCESSOR ON TRAINING DATA
    # ==================================================

    print("\nFitting preprocessing on training data...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # ==================================================
    # 12. TRANSFORM TEST DATA
    # ==================================================

    print("Transforming testing data...")

    X_test_processed = preprocessor.transform(
        X_test
    )

    # ==================================================
    # 13. FINAL RESULTS
    # ==================================================

    print("\n========== PREPROCESSING COMPLETED ==========")

    print("\nOriginal dataset shape:")
    print(data.shape)

    print("\nBefore preprocessing:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    print("\nAfter preprocessing:")

    print(
        "X_train_processed:",
        X_train_processed.shape
    )

    print(
        "X_test_processed :",
        X_test_processed.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "y_test :",
        y_test.shape
    )

    print("\n========== PREPROCESSING SUCCESSFUL ==========")

    # ==================================================
    # 14. RETURN RESULTS
    # ==================================================

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# ======================================================
# RUN PROGRAM
# ======================================================

if __name__ == "__main__":

    preprocess_data()
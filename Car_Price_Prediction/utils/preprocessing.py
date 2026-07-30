# =============================================================
# utils/preprocessing.py
# Purpose: All data preprocessing and feature engineering steps.
#          Handles loading, cleaning, encoding, and splitting.
# =============================================================

import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# ---------------------------------------------
# 1. Load Data
# ---------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the car.csv file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded successfully.")
    print(f"       Shape : {df.shape[0]} rows x {df.shape[1]} columns\n")
    return df


# ---------------------------------------------
# 2. Basic Data Inspection
# ---------------------------------------------
def inspect_data(df: pd.DataFrame) -> None:
    """
    Print an overview of the dataset:
    - First 5 rows
    - Last 5 rows
    - Data types and non-null counts
    - Descriptive statistics
    """
    print("=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head().to_string(), "\n")

    print("=" * 60)
    print("LAST 5 ROWS")
    print("=" * 60)
    print(df.tail().to_string(), "\n")

    print("=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    df.info()
    print()

    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(df.describe(include="all").to_string(), "\n")


# ---------------------------------------------
# 3. Check Missing Values
# ---------------------------------------------
def check_missing(df: pd.DataFrame) -> pd.Series:
    """
    Detect and print missing values per column.

    Returns
    -------
    pd.Series
        Count of null values per column.
    """
    missing = df.isnull().sum()
    print("=" * 60)
    print("MISSING VALUES PER COLUMN")
    print("=" * 60)
    print(missing.to_string(), "\n")
    return missing


# ---------------------------------------------
# 4. Check and Remove Duplicates
# ---------------------------------------------
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and remove duplicate rows from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with duplicates removed.
    """
    n_dups = df.duplicated().sum()
    print(f"[INFO] Duplicate rows found : {n_dups}")

    if n_dups > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[INFO] Duplicates removed. New shape : {df.shape}")

    return df


# ---------------------------------------------
# 5. Feature Engineering
# ---------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features and remove irrelevant columns.

    Steps:
      - Drop 'Car_Name' (high cardinality, not useful as-is)
      - Create 'Car_Age' = Current Year - Year
      - Drop 'Year' column after creating Car_Age

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe.
    """
    current_year = datetime.datetime.now().year

    # Drop Car_Name - too many unique values for simple encoding
    if "Car_Name" in df.columns:
        df = df.drop(columns=["Car_Name"])
        print("[INFO] Dropped 'Car_Name' column.")

    # Create Car_Age feature
    df["Car_Age"] = current_year - df["Year"]
    print(f"[INFO] Created 'Car_Age' (Current Year {current_year} - Year).")

    # Drop original Year column
    df = df.drop(columns=["Year"])
    print("[INFO] Dropped 'Year' column.\n")

    return df


# ---------------------------------------------
# 6. OneHot Encode Categorical Variables
# ---------------------------------------------
def encode_categoricals(
    df: pd.DataFrame,
    encoder_save_path: str = "encoder.pkl",
    fit: bool = True,
    encoder=None,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    Apply OneHotEncoding to categorical columns:
    Fuel_Type, Seller_Type, Transmission.

    Parameters
    ----------
    df            : Input dataframe.
    encoder_save_path : Path to save the fitted encoder.
    fit           : If True, fit a new encoder; else transform with existing.
    encoder       : Pre-fitted encoder (used when fit=False).

    Returns
    -------
    (pd.DataFrame, OneHotEncoder)
        Encoded dataframe and the fitted encoder.
    """
    cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]

    if fit:
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        encoded_array = encoder.fit_transform(df[cat_cols])

        # Persist encoder for Flask app
        with open(encoder_save_path, "wb") as f:
            pickle.dump(encoder, f)
        print(f"[INFO] Encoder fitted and saved -> {encoder_save_path}")
    else:
        encoded_array = encoder.transform(df[cat_cols])

    # Build readable column names
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    # Drop originals and concat encoded
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)

    print(f"[INFO] Categorical encoding complete. Shape after encoding : {df.shape}\n")
    return df, encoder


# ---------------------------------------------
# 7. Split Features and Target
# ---------------------------------------------
def split_features_target(
    df: pd.DataFrame, target_col: str = "Selling_Price"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the DataFrame into independent variables (X)
    and the dependent/target variable (y).

    Parameters
    ----------
    df         : Encoded dataframe.
    target_col : Name of the target column.

    Returns
    -------
    (X, y) : Feature matrix and target series.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"[INFO] Features  (X) shape : {X.shape}")
    print(f"[INFO] Target    (y) shape : {y.shape}\n")
    return X, y


# ---------------------------------------------
# 8. Train-Test Split
# ---------------------------------------------
def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split X and y into training and test sets (80/20 split).

    Parameters
    ----------
    X            : Feature matrix.
    y            : Target series.
    test_size    : Fraction of data for test set (default 0.20).
    random_state : Reproducibility seed.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train size : {X_train.shape[0]} samples")
    print(f"[INFO] Test  size : {X_test.shape[0]} samples\n")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------
# 9. Full Preprocessing Pipeline
# ---------------------------------------------
def run_preprocessing_pipeline(
    filepath: str,
    encoder_save_path: str = "encoder.pkl",
) -> tuple:
    """
    Convenience function that runs every preprocessing step
    in the correct order.

    Returns
    -------
    (X_train, X_test, y_train, y_test, df_encoded, encoder)
    """
    # Step 1 - Load
    df = load_data(filepath)

    # Step 2 - Inspect
    inspect_data(df)

    # Step 3 - Missing values
    check_missing(df)

    # Step 4 - Duplicates
    df = handle_duplicates(df)

    # Step 5 - Feature engineering
    df = feature_engineering(df)

    # Step 6 - Encode categoricals
    df, encoder = encode_categoricals(df, encoder_save_path=encoder_save_path)

    # Step 7 - Split X / y
    X, y = split_features_target(df)

    # Step 8 - Train / Test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    return X_train, X_test, y_train, y_test, df, encoder

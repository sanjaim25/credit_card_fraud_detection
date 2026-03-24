"""
data_preprocessing.py
---------------------
Functions for loading, cleaning, preprocessing,
and saving the credit card fraud dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw dataset.

    Parameters
    ----------
    filepath : str
        Path to CSV file

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print(f" Dataset loaded successfully")
        print(f" Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    except Exception as e:
        print(" Error loading dataset:", e)
        raise


def check_missing(df: pd.DataFrame) -> pd.Series:
    """
    Check missing values in dataset.

    Returns
    -------
    pd.Series
        Missing value count per column
    """
    missing = df.isnull().sum()

    print("\n Missing Values Per Column")
    print(missing)

    print(f"\n Total Missing Values: {missing.sum()}")

    return missing


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from dataset.
    """
    before = df.shape[0]

    df = df.drop_duplicates()

    after = df.shape[0]

    print(f"\n Removed {before - after} duplicate rows")

    return df


def scale_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Standard scale numeric columns.

    Parameters
    ----------
    df : DataFrame
    columns : list
        Columns to scale

    Returns
    -------
    DataFrame
    """
    scaler = StandardScaler()

    df[columns] = scaler.fit_transform(df[columns])

    print(f"\n Scaled Columns: {columns}")

    return df


def get_class_stats(df: pd.DataFrame, target: str = "Class") -> dict:
    """
    Compute fraud statistics.

    Returns
    -------
    dict
    """
    counts = df[target].value_counts()

    legit = int(counts.get(0, 0))
    fraud = int(counts.get(1, 0))

    fraud_pct = (fraud / len(df)) * 100

    stats = {
        "legitimate_transactions": legit,
        "fraud_transactions": fraud,
        "fraud_percentage": round(fraud_pct, 4),
    }

    print("\n Fraud Statistics")
    print(stats)

    return stats


def save_processed(df: pd.DataFrame, filepath: str) -> None:
    """
    Save processed dataset to CSV.
    """
    df.to_csv(filepath, index=False)

    print(f"\n Processed dataset saved to: {filepath}")
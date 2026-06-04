import pandas as pd

def dataset_health(df: pd.DataFrame):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    health_score = 100 - ((missing_cells / total_cells) * 100)

    return {
        "shape": df.shape,
        "missing_cells": int(missing_cells),
        "duplicate_rows": int(duplicate_rows),
        "health_score": round(health_score, 2)
    }


def handle_missing_values(df: pd.DataFrame, strategy="Mean"):
    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns
    all_cols = df.columns

    if strategy == "Mean":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())

    elif strategy == "Median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    elif strategy == "Mode":
        for col in all_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna("Unknown")

    elif strategy == "Drop Rows":
        df = df.dropna()

    return df


def remove_duplicates(df: pd.DataFrame, keep="first"):
    return df.drop_duplicates(keep=keep)


def detect_outliers_iqr(df: pd.DataFrame, column: str, return_filtered=False):
    df = df.copy()

    if column not in df.columns:
        return None

    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    result = {
        "column": column,
        "outlier_count": len(outliers),
        "lower_bound": lower,
        "upper_bound": upper
    }

    if return_filtered:
        filtered_df = df[(df[column] >= lower) & (df[column] <= upper)]
        return result, filtered_df

    return result
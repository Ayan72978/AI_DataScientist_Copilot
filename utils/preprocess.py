import pandas as pd

def dataset_health(df):

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()

    return 100 - ((missing_cells / total_cells) * 100)

def handle_missing_values(df, strategy):

    numeric_cols = df.select_dtypes(include="number").columns

    if strategy == "Mean":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())

    elif strategy == "Median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

    elif strategy == "Mode":
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    elif strategy == "Drop Rows":
        df = df.dropna()

    return df

def remove_duplicates(df, keep="first"):

    return df.drop_duplicates(keep=keep)

def detect_outliers_iqr(df, column):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    return outliers
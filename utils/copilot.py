import pandas as pd


def dataset_summary(df):

    summary = {}

    summary["rows"] = df.shape[0]
    summary["columns"] = df.shape[1]

    summary["missing_values"] = int(df.isnull().sum().sum())
    summary["duplicate_rows"] = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    summary["numeric_columns"] = numeric_cols
    summary["categorical_columns"] = categorical_cols

    total_cells = df.shape[0] * df.shape[1]

    if total_cells > 0:
        quality_score = (
            1 - (summary["missing_values"] / total_cells)
        ) * 100
    else:
        quality_score = 0

    summary["quality_score"] = round(
        quality_score,
        2
    )

    return summary


def suggest_target_column(df):

    target = df.columns[-1]

    max_unique = 0

    for col in df.columns:

        if df[col].dtype == "object":
            continue

        unique_count = df[col].nunique()

        if unique_count > max_unique:
            max_unique = unique_count
            target = col

    return target

def detect_problem_type(df, target):
    if df[target].dtype == "object" or df[target].nunique() < 10:
        return "Classification"
    else:
        return "Regression"


def suggest_model(df, target):

    if target not in df.columns:
        return "Unknown"

    if df[target].dtype == "object":

        unique_values = df[target].nunique()

        if unique_values <= 10:
            return "Random Forest Classifier"
        else:
            return "Logistic Regression"

    else:

        if df.shape[0] < 1000:
            return "Random Forest Regressor"
        else:
            return "Linear Regression"


def generate_insights(df):

    insights = []

    insights.append(
        f"Dataset contains {df.shape[0]} rows."
    )

    insights.append(
        f"Dataset contains {df.shape[1]} columns."
    )

    missing = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    insights.append(
        f"Missing values detected: {missing}"
    )

    insights.append(
        f"Duplicate rows detected: {duplicates}"
    )

    if missing == 0:
        insights.append(
            "Dataset has no missing values."
        )

    if duplicates == 0:
        insights.append(
            "Dataset has no duplicate rows."
        )

    if df.shape[0] < 100:
        insights.append(
            "Small dataset detected. Risk of overfitting."
        )

    if len(
        df.select_dtypes(include=["object"]).columns
    ) > 5:
        insights.append(
            "Many categorical features detected."
        )

    return insights


def next_action_plan(df, target):

    actions = []

    missing = int(
        df.isnull().sum().sum()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    if missing > 0:
        actions.append(
            "Handle missing values."
        )

    if duplicates > 0:
        actions.append(
            "Remove duplicate rows."
        )

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numeric_cols) < 2:
        actions.append(
            "Add more numeric features."
        )

    if df.shape[0] < 100:
        actions.append(
            "Collect more data to improve model performance."
        )

    if target in df.columns:

        if df[target].dtype == "object":
            actions.append(
                "Classification problem detected."
            )
        else:
            actions.append(
                "Regression problem detected."
            )

    actions.append(
        "Perform exploratory data analysis."
    )

    actions.append(
        "Train and compare multiple ML models."
    )

    actions.append(
        "Evaluate model performance before deployment."
    )

    return actions



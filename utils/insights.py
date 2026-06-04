def generate_insights(df):

    insights = []

    insights.append(
        f"Dataset contains {df.shape[0]} rows."
    )

    insights.append(
        f"Dataset contains {df.shape[1]} columns."
    )

    missing = df.isnull().sum().sum()

    insights.append(
        f"Missing values: {missing}"
    )

    duplicates = df.duplicated().sum()

    insights.append(
        f"Duplicate rows: {duplicates}"
    )

    return insights
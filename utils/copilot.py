import pandas as pd


# ==========================================================
# DATASET SUMMARY
# ==========================================================

def dataset_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }


# ==========================================================
# TARGET COLUMN
# ==========================================================

def suggest_target_column(df):

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols):
        return numeric_cols[-1]

    return df.columns[-1]


# ==========================================================
# PROBLEM TYPE
# ==========================================================

def detect_problem_type(df, target):

    if df[target].dtype == "object":
        return "Classification"

    if df[target].nunique() < 10:
        return "Classification"

    return "Regression"


# ==========================================================
# MODEL SUGGESTION
# ==========================================================

def suggest_model(df, target):

    problem = detect_problem_type(df, target)

    if problem == "Classification":
        return "Random Forest Classifier"

    return "Random Forest Regressor"


# ==========================================================
# QUALITY SCORE
# ==========================================================

def data_quality_score(df):

    total = df.shape[0] * df.shape[1]

    if total == 0:
        return 0

    missing = df.isnull().sum().sum()

    score = 100 - (missing / total * 100)

    return round(score)


# ==========================================================
# INSIGHTS
# ==========================================================

def generate_insights(df):

    insights = []

    insights.append(
        f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."
    )

    missing = int(df.isnull().sum().sum())

    if missing:
        insights.append(
            f"Found {missing} missing values."
        )
    else:
        insights.append(
            "No missing values found."
        )

    duplicates = int(df.duplicated().sum())

    if duplicates:
        insights.append(
            f"Found {duplicates} duplicate rows."
        )

    numeric_cols = len(
        df.select_dtypes(include="number").columns
    )

    insights.append(
        f"Dataset contains {numeric_cols} numeric columns."
    )

    return insights


# ==========================================================
# ACTION PLAN
# ==========================================================

def next_action_plan(df, target):

    actions = []

    if df.isnull().sum().sum():
        actions.append(
            "Handle missing values."
        )

    if df.duplicated().sum():
        actions.append(
            "Remove duplicate rows."
        )

    actions.extend([
        "Perform EDA.",
        "Create visualizations.",
        "Train ML models.",
        "Evaluate performance.",
        "Deploy best model."
    ])

    return actions


# ==========================================================
# SMART CHAT
# ==========================================================

def smart_chat(question, df):

    q = question.lower().strip()

    target = suggest_target_column(df)

    results = []

    # --------------------------
    # Rows
    # --------------------------

    if any(x in q for x in ["row", "rows"]):
        results.append(
            f"📊 Total Rows: {df.shape[0]}"
        )

    # --------------------------
    # Columns
    # --------------------------

    if any(x in q for x in ["column", "columns"]):
        results.append(
            f"📊 Total Columns: {df.shape[1]}"
        )

    # --------------------------
    # Shape
    # --------------------------

    if any(x in q for x in ["shape", "size"]):
        results.append(
            f"📦 Dataset Shape: {df.shape}"
        )

    # --------------------------
    # Missing Values
    # --------------------------

    if any(x in q for x in ["missing", "null", "nan"]):

        mv = df.isnull().sum()
        mv = mv[mv > 0]

        if len(mv) == 0:

            results.append(
                "✅ No missing values found."
            )

        else:

            results.append(
                "⚠️ Missing Values:"
            )

            for col, val in mv.items():

                results.append(
                    f"{col}: {val}"
                )

    # --------------------------
    # Duplicates
    # --------------------------

    if any(x in q for x in [
        "duplicate",
        "duplicates"
    ]):

        results.append(
            f"📄 Duplicate Rows: {int(df.duplicated().sum())}"
        )

    # --------------------------
    # Quality Score
    # --------------------------

    if any(x in q for x in [
        "quality",
        "score"
    ]):

        results.append(
            f"✅ Quality Score: {data_quality_score(df)}/100"
        )

    # --------------------------
    # Target
    # --------------------------

    if any(x in q for x in [
        "target",
        "label"
    ]):

        results.extend([
            f"🎯 Target Column: {target}",
            f"🤖 Problem Type: {detect_problem_type(df,target)}",
            f"🧠 Suggested Model: {suggest_model(df,target)}"
        ])

    # --------------------------
    # Insights
    # --------------------------

    if any(x in q for x in [
        "insight",
        "insights"
    ]):

        results.append(
            "💡 Dataset Insights:"
        )

        results.extend(
            generate_insights(df)
        )

    # --------------------------
    # Action Plan
    # --------------------------

    if any(x in q for x in [
        "action",
        "plan",
        "next step",
        "next steps"
    ]):

        actions = next_action_plan(
            df,
            target
        )

        results.append(
            "🚀 Action Plan:"
        )

        for i, step in enumerate(actions, 1):

            results.append(
                f"Step {i}: {step}"
            )

    # --------------------------
    # Correlation
    # --------------------------

    if any(x in q for x in [
        "correlation",
        "correlations",
        "relationship"
    ]):

        numeric = df.select_dtypes(
            include="number"
        )

        if len(numeric.columns) < 2:

            results.append(
                "Not enough numeric columns."
            )

        else:

            corr = numeric.corr().abs()

            pairs = []

            for c1 in corr.columns:
                for c2 in corr.columns:

                    if c1 < c2:

                        pairs.append(
                            (
                                c1,
                                c2,
                                corr.loc[c1, c2]
                            )
                        )

            pairs.sort(
                key=lambda x: x[2],
                reverse=True
            )

            results.append(
                "📈 Top Correlations:"
            )

            for c1, c2, value in pairs[:5]:

                results.append(
                    f"{c1} ↔ {c2}: {round(value,3)}"
                )

    # --------------------------
    # Column Analysis
    # --------------------------

    for col in df.columns:

        if col.lower() in q:

            results.append("")
            results.append(
                f"📌 Analysis of {col}"
            )

            results.append(
                f"Type: {df[col].dtype}"
            )

            results.append(
                f"Unique Values: {df[col].nunique()}"
            )

            results.append(
                f"Missing Values: {df[col].isnull().sum()}"
            )

            if pd.api.types.is_numeric_dtype(df[col]):

                results.extend([
                    f"Mean: {round(df[col].mean(),2)}",
                    f"Median: {round(df[col].median(),2)}",
                    f"Min: {round(df[col].min(),2)}",
                    f"Max: {round(df[col].max(),2)}"
                ])

            else:

                mode = df[col].mode()

                if len(mode):

                    results.append(
                        f"Top Value: {mode.iloc[0]}"
                    )

    # --------------------------
    # Return Results
    # --------------------------

    if results:
        return results

    return [
        "🤖 I understand your dataset.",
        "",
        "Try asking:",
        "- rows and columns",
        "- shape",
        "- missing values",
        "- duplicates",
        "- target column",
        "- quality score",
        "- insights",
        "- action plan",
        "- correlations",
        "- analyze age",
        "- analyze salary"
    ]
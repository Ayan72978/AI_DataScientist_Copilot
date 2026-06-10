import streamlit as st
import pandas as pd
import joblib

from utils.copilot import (
    dataset_summary,
    suggest_target_column,
    suggest_model,
    next_action_plan
)

from utils.preprocess import (
    dataset_health,
    handle_missing_values,
    remove_duplicates,
    detect_outliers_iqr
)

from utils.visualize import (
    histogram,
    boxplot,
    scatter,
    correlation_heatmap,
    missing_heatmap,
    pair_plot,
    count_plot,
    pie_chart,
    line_chart,
    distribution_plot,
    violin_plot,
    bar_plot
)

from utils.insights import generate_insights

from utils.ml_models import (
    train_models,
    auto_ml_pipeline,
    predict_single,
    get_feature_importance
)

st.set_page_config(
    page_title="AI Data Scientist Copilot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Scientist Copilot")
st.markdown("### Made by Ayan Behlim")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data Cleaning",
        "Visualizations",
        "AI Insights",
        "Machine Learning",
        "Copilot"
    ]
)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    st.session_state["df"] = pd.read_csv(uploaded_file)

df = st.session_state.get("df", None)

if df is None:
    st.info("👆 Upload a CSV file to start")
    st.stop()

# =========================
# OVERVIEW
# =========================
if page == "Overview":

    st.header("📊 Dataset Overview")

    st.success("🚀 Welcome to AI Data Scientist Copilot")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicates", int(df.duplicated().sum()))

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.divider()

    health = dataset_health(df)

    st.metric("Health Score", f"{health['health_score']}%")
    st.progress(int(health["health_score"]))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Numeric Features",
            len(df.select_dtypes(include="number").columns)
        )

    with col2:
        st.metric(
            "Categorical Features",
            len(df.select_dtypes(include=["object"]).columns)
        )


# =========================
# DATA CLEANING
# =========================
elif page == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    strategy = st.selectbox(
        "Missing Value Strategy",
        ["Mean", "Median", "Mode", "Drop Rows"]
    )

    if st.button("Apply Cleaning"):

        df_cleaned = handle_missing_values(df.copy(), strategy)
        df_cleaned = remove_duplicates(df_cleaned)

        st.session_state["df"] = df_cleaned
        df = df_cleaned

        st.success("Cleaning completed successfully")

    st.dataframe(df.head(), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Cleaned Dataset",
        csv,
        "cleaned_dataset.csv",
        "text/csv"
    )


    
# =========================
# VISUALIZATIONS
# =========================

elif page == "Visualizations":

    st.header("📈 Advanced Visualizations")

    graph_type = st.selectbox(
        "Select Visualization",
        [
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "Correlation Heatmap",
            "Missing Values Heatmap",
            "Pair Plot",
            "Count Plot",
            "Pie Chart",
            "Line Chart",
            "Distribution Plot",
            "Violin Plot",
            "Bar Plot"
        ]
    )

    numeric_cols = list(
        df.select_dtypes(
            include="number"
        ).columns
    )

    cat_cols = list(
        df.select_dtypes(
            include=["object", "category"]
        ).columns
    )

    if graph_type == "Histogram" and numeric_cols:
        col = st.selectbox(
            "Column",
            numeric_cols
        )
        histogram(df, col)

    elif graph_type == "Box Plot" and numeric_cols:
        col = st.selectbox(
            "Column",
            numeric_cols
        )
        boxplot(df, col)

    elif graph_type == "Scatter Plot" and len(numeric_cols) >= 2:
        x = st.selectbox(
            "X Axis",
            numeric_cols
        )
        y = st.selectbox(
            "Y Axis",
            numeric_cols
        )
        scatter(df, x, y)

    elif graph_type == "Correlation Heatmap":
        correlation_heatmap(df)

    elif graph_type == "Missing Values Heatmap":
        missing_heatmap(df)

    elif graph_type == "Pair Plot":
        pair_plot(df)

    elif graph_type == "Count Plot" and cat_cols:
        col = st.selectbox(
            "Column",
            cat_cols
        )
        count_plot(df, col)

    elif graph_type == "Pie Chart" and cat_cols:
        col = st.selectbox(
            "Column",
            cat_cols
        )
        pie_chart(df, col)

    elif graph_type == "Line Chart" and numeric_cols:
        col = st.selectbox(
            "Column",
            numeric_cols
        )
        line_chart(df, col)

    elif graph_type == "Distribution Plot" and numeric_cols:
        col = st.selectbox(
            "Column",
            numeric_cols
        )
        distribution_plot(df, col)

    elif graph_type == "Violin Plot" and numeric_cols:
        col = st.selectbox(
            "Column",
            numeric_cols
        )
        violin_plot(df, col)

    elif graph_type == "Bar Plot" and cat_cols:
        col = st.selectbox(
            "Column",
            cat_cols
        )
        bar_plot(df, col)

# =========================
# AI INSIGHTS
# =========================

elif page == "AI Insights":

    st.header("🧠 AI Insights")

    insights = generate_insights(df)

    for insight in insights:
        st.success(
            str(insight)
        )

    st.divider()

    st.subheader(
        "Dataset Summary"
    )

    st.write(
        f"Rows: {df.shape[0]}"
    )

    st.write(
        f"Columns: {df.shape[1]}"
    )

    st.write(
        f"Missing Values: {df.isnull().sum().sum()}"
    )

    st.write(
        f"Duplicate Rows: {df.duplicated().sum()}"
    )

# =========================
# MACHINE LEARNING
# =========================

elif page == "Machine Learning":

    st.header("🤖 Machine Learning Engine")

    target = st.selectbox(
        "Select Target Column",
        df.columns
    )

    if st.button("Train Models"):

        df_ml = df.copy().dropna()

        if len(df_ml) < 10:

            st.error(
                "Not enough data for training"
            )

            st.stop()

        for col in df_ml.columns:

            if df_ml[col].dtype == "object":

                df_ml[col] = pd.factorize(
                    df_ml[col]
                )[0]

        X = df_ml.drop(
            columns=[target]
        )

        y = df_ml[target]

        (
            best_model,
            best_name,
            results,
            X_test,
            y_test,
            problem
        ) = train_models(
            X,
            y
        )

        st.subheader(
            "🏆 Model Results"
        )

        result_df = pd.DataFrame(
            list(results.items()),
            columns=[
                "Model",
                "Score"
            ]
        )

        st.dataframe(
            result_df,
            use_container_width=True
        )

        st.subheader(
            "📊 Model Comparison"
        )

        chart_df = result_df.set_index(
            "Model"
        )

        st.bar_chart(
            chart_df
        )

        st.success(
            f"Best Model: {best_name}"
        )

        st.info(
            f"Problem Type: {problem}"
        )

        st.session_state["model"] = best_model

        st.session_state["columns"] = X.columns

        joblib.dump(
            best_model,
            "best_model.pkl"
        )

        with open(
            "best_model.pkl",
            "rb"
        ) as file:

            st.download_button(
                "📥 Download Model",
                file,
                file_name="best_model.pkl"
            )

        try:

            importance = get_feature_importance(
                best_model,
                X.columns
            )

            if importance:

                st.subheader(
                    "📊 Feature Importance"
                )

                importance_df = pd.DataFrame(
                    importance,
                    columns=[
                        "Feature",
                        "Importance"
                    ]
                )

                st.dataframe(
                    importance_df,
                    use_container_width=True
                )

                st.subheader(
                    "📈 Feature Importance Chart"
                )

                feature_chart = importance_df.set_index(
                    "Feature"
                )

                st.bar_chart(
                    feature_chart
                )

        except Exception:

            pass

    if "model" in st.session_state:

        st.divider()

        st.subheader(
            "🔮 Prediction System"
        )

        model = st.session_state["model"]

        columns = st.session_state["columns"]

        input_data = []

        for col in columns:

            value = st.number_input(
                col,
                value=0.0
            )

            input_data.append(
                value
            )

        if st.button(
            "Predict"
        ):

            prediction = predict_single(
                model,
                input_data
            )

            st.success(
                f"Prediction: {prediction}"
            )

            if "prediction_history" not in st.session_state:

                st.session_state[
                    "prediction_history"
                ] = []

            st.session_state[
                "prediction_history"
            ].append(
                prediction
            )

        if "prediction_history" in st.session_state:

            st.subheader(
                "📜 Prediction History"
            )

            history_df = pd.DataFrame(
                st.session_state[
                    "prediction_history"
                ],
                columns=[
                    "Prediction"
                ]
            )

            st.dataframe(
                history_df,
                use_container_width=True
            )

            csv = history_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )


    
# =========================
# COPILOT
# =========================

elif page == "Copilot":

    st.header("🧠 AI Copilot Dashboard")

    summary = dataset_summary(df)
    target = suggest_target_column(df)
    problem = detect_problem_type(df, target)
    score = data_quality_score(df)
    
    model = suggest_model(
        df,
        target
    )

    insights = generate_insights(df)

    actions = next_action_plan(
        df,
        target
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", summary["rows"])
    col2.metric("Columns", summary["columns"])
    col3.metric("Missing", summary["missing_values"])

    st.divider()

    st.subheader("🤖 Recommendations")

    st.success(f"Suggested Target: {target}")
    st.info(f"Suggested Model: {model}")
    st.subheader("🧠 AI Problem Detection")
    st.info(f"Detected Problem Type: {problem}")
    
    st.subheader("📊 AI Data Quality Score")

    st.metric("Score", f"{score}/100")
    st.progress(int(score))

    st.subheader("🎯 Dataset Recommendation")

    if df.shape[0] < 100:
        st.warning("Small dataset detected. More data may improve model performance.")

    elif df.shape[0] < 1000:
        st.info("Medium-sized dataset detected. Suitable for most machine learning algorithms.")

    else:
        st.success("Large dataset detected. Excellent for advanced machine learning models.")

    st.subheader("📋 Dataset Quality Report")

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    if missing == 0:
        st.success("No missing values detected.")
    else:
        st.warning(f"{missing} missing values detected.")

    if duplicates == 0:
        st.success("No duplicate rows detected.")
    else:
        st.warning(f"{duplicates} duplicate rows detected.")

    report = f"""
Dataset Summary

Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing Values: {missing}
Duplicate Rows: {duplicates}

Suggested Target: {target}
Suggested Model: {model}
"""

    st.download_button(
        label="📥 Download Summary Report",
        data=report,
        file_name="dataset_report.txt",
        mime="text/plain"
    )

    st.divider()

    st.subheader("💡 Smart Insights")

    for insight in insights:
        st.success(str(insight))

    st.divider()

    st.subheader("🚀 Action Plan")

    for action in actions:
        st.warning(action)

    st.divider()

    st.subheader("💬 Ask Your Dataset")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    question = st.text_input("Ask a question")

    if question:

        q = question.lower()
        answer = ""

        if "row" in q:
            answer = f"Dataset contains {df.shape[0]} rows"

        elif "column" in q:
            answer = f"Dataset contains {df.shape[1]} columns"

        elif "missing" in q:
            answer = f"Missing values: {df.isnull().sum().sum()}"

        elif "duplicate" in q:
            answer = f"Duplicate rows: {df.duplicated().sum()}"

        elif "target" in q:
            answer = f"Suggested target: {target}"

        elif "model" in q:
            answer = f"Suggested model: {model}"

        else:
            answer = "Try asking about rows, columns, missing values, target column, or model recommendation."

        st.session_state["chat_history"].append((question, answer))

    st.markdown("### Chat History")

    for q, a in reversed(st.session_state["chat_history"][-10:]):
        st.markdown(f"**You:** {q}")
        st.success(a)
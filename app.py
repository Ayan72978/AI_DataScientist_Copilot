import streamlit as st
import pandas as pd
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from utils.automl import train_automl
from utils.reports import generate_report

from utils.copilot import (
    dataset_summary,
    suggest_target_column,
    suggest_model,
    generate_insights,
    next_action_plan,
    detect_problem_type,
    data_quality_score
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

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🧹 Data Cleaning",
        "📊 Visualization",
        "🧠 AI Insights",
        "🤖 Copilot",
        "⚙ Auto ML",
        "🔮 Predict Data",
        "📄 Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("ML + AI Project Dashboard")

st.title("🤖 AI Data Scientist Copilot")

# -------------------------
# HEADER
# -------------------------
st.header(page)

st.caption(
    "Upload data, clean it, visualize insights, train ML models and make predictions."
)

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "📁 Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

# -------------------------
# LOAD DATASET
# -------------------------
if uploaded_file is not None:
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")

        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        else:
            st.error("❌ Unsupported file format")
            st.stop()

        st.session_state["df"] = df

        st.success(
            f"✅ Dataset loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns)"
        )

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

# -------------------------
# SAFE DATA ACCESS
# -------------------------
df = st.session_state.get("df", None)

if df is None:
    st.info("📁 Please upload a dataset to continue.")
    st.stop()


# =========================
# 🏠 OVERVIEW
# =========================
if page == "🏠 Overview":

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
# 🧹 DATA CLEANING
# =========================
elif page == "🧹 Data Cleaning":

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
# 📊 VISUALIZATION
# =========================
elif page == "📊 Visualization":

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

    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    if graph_type == "Histogram" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        histogram(df, col)

    elif graph_type == "Box Plot" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        boxplot(df, col)

    elif graph_type == "Scatter Plot" and len(numeric_cols) >= 2:
        x = st.selectbox("X Axis", numeric_cols)
        y = st.selectbox("Y Axis", numeric_cols)
        scatter(df, x, y)

    elif graph_type == "Correlation Heatmap":
        correlation_heatmap(df)

    elif graph_type == "Missing Values Heatmap":
        missing_heatmap(df)

    elif graph_type == "Pair Plot":
        pair_plot(df)

    elif graph_type == "Count Plot" and cat_cols:
        col = st.selectbox("Column", cat_cols)
        count_plot(df, col)

    elif graph_type == "Pie Chart" and cat_cols:
        col = st.selectbox("Column", cat_cols)
        pie_chart(df, col)

    elif graph_type == "Line Chart" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        line_chart(df, col)

    elif graph_type == "Distribution Plot" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        distribution_plot(df, col)

    elif graph_type == "Violin Plot" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        violin_plot(df, col)

    elif graph_type == "Bar Plot" and cat_cols:
        col = st.selectbox("Column", cat_cols)
        bar_plot(df, col)


# =========================
# 🧠 AI INSIGHTS
# =========================
elif page == "🧠 AI Insights":

    insights = generate_insights(df)

    for insight in insights:
        st.success(str(insight))

    st.divider()

    st.subheader("Dataset Summary")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"Missing Values: {df.isnull().sum().sum()}")
    st.write(f"Duplicate Rows: {df.duplicated().sum()}")


# =========================
# 🤖 COPILOT
# =========================
elif page == "🤖 Copilot":

    summary = dataset_summary(df)
    target = suggest_target_column(df)
    problem = detect_problem_type(df, target)
    score = data_quality_score(df)
    model = suggest_model(df, target)
    insights = generate_insights(df)
    actions = next_action_plan(df, target)

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
Problem Type: {problem}
Quality Score: {score}/100
"""

    st.download_button(
        label="📥 Download Summary Report",
        data=report,
        file_name="dataset_report.txt",
        mime="text/plain"
    )

    report_df = pd.DataFrame(
        {
            "Metric": [
                "Rows", "Columns", "Missing Values", "Duplicate Rows",
                "Target Column", "Suggested Model", "Problem Type", "Quality Score"
            ],
            "Value": [
                df.shape[0], df.shape[1], missing, duplicates,
                target, model, problem, score
            ]
        }
    )

    csv_report = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📊 Download Detailed Report",
        data=csv_report,
        file_name="ai_copilot_report.csv",
        mime="text/csv"
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


# =========================
# ⚙ AUTO ML
# =========================
elif page == "⚙ Auto ML":

    target = suggest_target_column(df)

    if target is None:
        st.warning("No suitable target column found.")

    else:
        problem = detect_problem_type(df, target)

        st.write(f"🎯 Target Column: {target}")
        st.write(f"📌 Problem Type: {problem}")
        st.info("AutoML will train multiple models and select the best one.")

        if st.button("🚀 Train Models"):

            with st.spinner("Training models..."):
                (
                    results,
                    best_model_name,
                    best_model,
                    feature_columns
                ) = train_automl(df, target, problem)

            if len(results) == 0:
                st.error("❌ No model could be trained.")

            else:
                st.success(f"🏆 Best Model: {best_model_name}")

                leaderboard = pd.DataFrame(
                    {
                        "Model": list(results.keys()),
                        "Score": list(results.values())
                    }
                )

                leaderboard = leaderboard.sort_values(by="Score", ascending=False)

                st.subheader("🏆 Model Leaderboard")
                st.dataframe(leaderboard, use_container_width=True)

                st.subheader("📊 Model Performance")
                st.bar_chart(leaderboard.set_index("Model"))

                joblib.dump(best_model, "best_model.pkl")

                st.session_state["trained_models"] = True
                st.session_state["best_model"] = best_model
                st.session_state["best_model_name"] = best_model_name
                st.session_state["feature_columns"] = feature_columns

                st.success("✅ Model saved successfully.")

                st.subheader("📋 Features Used")
                st.write(feature_columns)

                csv = leaderboard.to_csv(index=False)

                st.download_button(
                    label="📥 Download Leaderboard",
                    data=csv,
                    file_name="model_leaderboard.csv",
                    mime="text/csv"
                )


# =========================
# 🔮 PREDICT DATA
# =========================
elif page == "🔮 Predict Data":

    if "best_model" not in st.session_state:
        st.warning("⚠ Please train a model first in Auto ML page")
        st.stop()

    if "feature_columns" not in st.session_state:
        st.warning("⚠ Feature columns not found. Please retrain model.")
        st.stop()

    model = st.session_state["best_model"]
    feature_cols = st.session_state["feature_columns"]

    predict_file = st.file_uploader(
        "Upload file for prediction",
        type=["csv", "xlsx", "xls"]
    )

    if predict_file is not None:

        try:
            file_name = predict_file.name.lower()

            if file_name.endswith(".csv"):
                try:
                    new_df = pd.read_csv(predict_file, encoding="utf-8")
                except Exception:
                    predict_file.seek(0)
                    new_df = pd.read_csv(predict_file, encoding="latin1")

            elif file_name.endswith((".xlsx", ".xls")):
                new_df = pd.read_excel(predict_file)

            else:
                st.error("❌ Unsupported file format")
                st.stop()

            st.subheader("📄 Input Data Preview")
            st.dataframe(new_df.head())

            X_new = new_df.reindex(columns=feature_cols)
            X_new = X_new.fillna(0)
            X_new = X_new.select_dtypes(include=["number"])

            if st.button("🚀 Predict"):

                predictions = model.predict(X_new)

                result_df = new_df.copy()
                result_df["Prediction"] = predictions

                st.subheader("📊 Prediction Results")
                st.dataframe(result_df)

                st.subheader("📈 Prediction Summary")
                st.write(result_df["Prediction"].value_counts())

                st.download_button(
                    "⬇ Download Predictions",
                    result_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")


# =========================
# 📄 REPORTS
# =========================
elif page == "📄 Reports":

    st.subheader("📄 Generate Report")

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    target = suggest_target_column(df)
    model_name = suggest_model(df, target)
    problem = detect_problem_type(df, target)
    score = data_quality_score(df)

    report_text = f"""
AI Data Scientist Copilot — Report
=====================================

Dataset Summary
---------------
Rows             : {df.shape[0]}
Columns          : {df.shape[1]}
Missing Values   : {missing}
Duplicate Rows   : {duplicates}

AI Recommendations
------------------
Suggested Target : {target}
Suggested Model  : {model_name}
Problem Type     : {problem}
Quality Score    : {score}/100

Column Info
-----------
{df.dtypes.to_string()}
"""

    st.text_area("Report Preview", report_text, height=300)

    st.download_button(
        label="📥 Download Report (.txt)",
        data=report_text,
        file_name="ai_report.txt",
        mime="text/plain"
    )

    report_df = pd.DataFrame(
        {
            "Metric": [
                "Rows", "Columns", "Missing Values", "Duplicate Rows",
                "Target Column", "Suggested Model", "Problem Type", "Quality Score"
            ],
            "Value": [
                df.shape[0], df.shape[1], missing, duplicates,
                target, model_name, problem, score
            ]
        }
    )

    csv_report = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📊 Download Report (.csv)",
        data=csv_report,
        file_name="ai_report.csv",
        mime="text/csv"
    )

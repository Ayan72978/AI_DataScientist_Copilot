import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
    pie_chart
)

from utils.insights import generate_insights

from utils.ml_models import train_models


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
    ["Overview", "Data Cleaning", "Visualizations", "AI Insights", "Machine Learning"]
)

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # =========================
    # OVERVIEW
    # =========================
    if page == "Overview":

        st.header("📊 Dataset Overview")

        st.write(df.head())

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        col4.metric("Duplicates", int(df.duplicated().sum()))

        health = dataset_health(df)

        st.subheader("Dataset Health")
        st.progress(int(health["health_score"]))
        st.success(f"Health Score: {health['health_score']}%")

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

            df = handle_missing_values(df.copy(), strategy)
            df = remove_duplicates(df)

            st.success("Data cleaned successfully")
            st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:

            col = st.selectbox("Outlier Column", numeric_cols)

            outliers = detect_outliers_iqr(df, col)

            if outliers is not None:

                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                outlier_df = df[(df[col] < lower) | (df[col] > upper)]

                st.metric("Outliers Found", len(outlier_df))
                st.dataframe(outlier_df.head())

    # =========================
    # VISUALIZATIONS
    # =========================
    elif page == "Visualizations":

        st.header("📈 Visualizations")

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:

            col = st.selectbox("Numeric Column", numeric_cols)

            histogram(df, col)
            boxplot(df, col)

        if len(numeric_cols) >= 2:

            x = st.selectbox("X Axis", numeric_cols, key="x")
            y = st.selectbox("Y Axis", numeric_cols, key="y")

            scatter(df, x, y)

        correlation_heatmap(df)
        missing_heatmap(df)

        if st.button("Pair Plot"):
            pair_plot(df)

        cat_cols = df.select_dtypes(include="object").columns

        if len(cat_cols) > 0:

            cat = st.selectbox("Categorical Column", cat_cols)

            count_plot(df, cat)
            pie_chart(df, cat)

    # =========================
    # AI INSIGHTS
    # =========================
    elif page == "AI Insights":

        st.header("🧠 AI Insights")

        insights = generate_insights(df)

        for i in insights:
            st.success(i)

    # =========================
    # MACHINE LEARNING (SAFE)
    # =========================
    elif page == "Machine Learning":

        st.header("🤖 AI Machine Learning Engine")

        target = st.selectbox("Select Target Column", df.columns)

        if st.button("🚀 Train Models"):

            df_ml = df.copy().dropna()

            if df_ml.shape[0] < 10:
                st.error("Not enough data after cleaning")
                st.stop()

            for col in df_ml.columns:
                if df_ml[col].dtype == "object":
                    df_ml[col] = pd.factorize(df_ml[col])[0]

            if target not in df_ml.columns:
                st.error("Target column issue")
                st.stop()

            X = df_ml.drop(columns=[target])
            y = df_ml[target]

            if X.shape[1] == 0:
                st.error("No feature columns available")
                st.stop()

            best_model, best_name, results, X_test, y_test = train_models(X, y)

            if best_model is None:
                st.error("Model training failed")
                st.stop()

            st.subheader("📊 Model Results")

            for name, acc in results.items():
                st.write(f"{name}: {acc:.4f}")

            st.success(f"🏆 Best Model: {best_name}")

            preds = best_model.predict(X_test)

            st.subheader("📉 Sample Predictions")
            st.write(preds[:10])

else:
    st.info("👆 Upload a CSV file to start")
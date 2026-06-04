import streamlit as st
import pandas as pd
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

from utils.ml_models import train_models, predict_single


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

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", int(df.isnull().sum().sum()))
        col4.metric("Duplicates", int(df.duplicated().sum()))

        st.subheader("Preview")
        st.dataframe(df.head())

        health = dataset_health(df)

        # SAFE FIX (progress bar)
        try:
            health_score = float(health)
            health_score = max(0, min(100, health_score))
            st.progress(int(health_score))
            st.success(f"Health Score: {health_score:.2f}%")
        except:
            st.warning("Health score cannot be calculated")

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

            st.success("Cleaning Done")
            st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:

            col = st.selectbox("Outlier Column", numeric_cols)

            outliers = detect_outliers_iqr(df, col)

            st.metric("Outliers Found", len(outliers))
            st.dataframe(outliers.head())

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
    # MACHINE LEARNING
    # =========================
    elif page == "Machine Learning":

        st.header("🤖 Machine Learning Engine")

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

            best_model, best_name, results, X_test, y_test = train_models(X, y)

            if best_model is None:
                st.error("Model training failed")
                st.stop()

            st.subheader("Model Results")

            for name, acc in results.items():
                st.write(f"{name}: {acc}")

            st.success(f"Best Model: {best_name}")

            # STORE MODEL SAFELY
            st.session_state["model"] = best_model
            st.session_state["columns"] = X.columns

        # =========================
        # PREDICTION SYSTEM (SAFE)
        # =========================
        if "model" in st.session_state:

            st.subheader("🔮 Prediction System")

            model = st.session_state["model"]
            cols = st.session_state["columns"]

            input_data = []

            for col in cols:
                val = st.number_input(col, value=0.0)
                input_data.append(val)

            if st.button("Predict"):

                try:
                    result = predict_single(model, input_data)
                    st.success(f"Prediction: {result}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

else:
    st.info("👆 Upload a CSV file to start")
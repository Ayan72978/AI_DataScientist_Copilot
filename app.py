import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

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

from utils.ml_models import (
    train_models,
    predict_single,
    save_model,
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

        st.download_button(
            "📥 Download Dataset",
            df.to_csv(index=False),
            "dataset.csv",
            "text/csv"
        )

        st.subheader("Dataset Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[col].dtype) for col in df.columns],
            "Missing": [df[col].isnull().sum() for col in df.columns]
        })

        st.dataframe(info)

        health = dataset_health(df)

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

        graph_type = st.selectbox(
            "Select Graph",
            [
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Correlation Heatmap",
                "Missing Values Heatmap",
                "Pair Plot",
                "Count Plot",
                "Pie Chart"
            ]
        )

        numeric_cols = list(df.select_dtypes(include="number").columns)
        cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

        if graph_type == "Histogram":
            if numeric_cols:
                col = st.selectbox("Select Column", numeric_cols)
                histogram(df, col)

        elif graph_type == "Box Plot":
            if numeric_cols:
                col = st.selectbox("Select Column", numeric_cols)
                boxplot(df, col)

        elif graph_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                x = st.selectbox("X Axis", numeric_cols)
                y = st.selectbox("Y Axis", numeric_cols)
                scatter(df, x, y)

        elif graph_type == "Correlation Heatmap":
            correlation_heatmap(df)

        elif graph_type == "Missing Values Heatmap":
            missing_heatmap(df)

        elif graph_type == "Pair Plot":
            pair_plot(df)

        elif graph_type == "Count Plot":
            if cat_cols:
                col = st.selectbox("Select Column", cat_cols)
                count_plot(df, col)

        elif graph_type == "Pie Chart":
            if cat_cols:
                col = st.selectbox("Select Column", cat_cols)
                pie_chart(df, col)

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

        if st.button("Train Models"):

            df_ml = df.copy().dropna()

            if df_ml.shape[0] < 10:
                st.error("Not enough data after cleaning")
                st.stop()

            for col in df_ml.columns:
                if df_ml[col].dtype == "object":
                    df_ml[col] = pd.factorize(df_ml[col])[0]

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

            # Save model
            joblib.dump(best_model, "best_model.pkl")

            st.download_button(
                "📥 Download Model",
                open("best_model.pkl", "rb"),
                file_name="best_model.pkl"
            )

            # Feature importance
            try:
                importance = get_feature_importance(best_model, X.columns)
                if importance is not None:
                    st.subheader("📊 Feature Importance")
                    st.dataframe(importance)
            except:
                pass

            st.session_state["model"] = best_model
            st.session_state["columns"] = X.columns

        # =========================
        # PREDICTION
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
import streamlit as st
import pandas as pd
import os
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

from utils.insights import (
    generate_insights
)

from utils.ml_models import (
    train_models
)

st.set_page_config(
    page_title="AI Data Scientist Copilot",
    page_icon="🤖",
    layout="wide"
)

st.sidebar.title("🤖 AI Copilot")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Cleaning",
        "Visualizations",
        "AI Insights",
        "Machine Learning"
    ]
)

st.title("🤖 AI Data Scientist Copilot")
st.markdown("### Made by Ayan Behlim")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

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

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Type"]))

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include="all"))

        health = dataset_health(df)

        st.subheader("Dataset Health")
        st.progress(int(health))
        st.success(f"Health Score: {health:.2f}%")

    # =========================
    # DATA CLEANING
    # =========================
    elif page == "Data Cleaning":

        st.header("🧹 Data Cleaning")

        strategy = st.selectbox(
            "Missing Value Strategy",
            ["Mean", "Median", "Mode", "Drop Rows"]
        )

        if st.button("Apply Missing Value Strategy"):
            cleaned_df = handle_missing_values(df.copy(), strategy)
            st.success("Strategy Applied Successfully")
            st.dataframe(cleaned_df.head())

        keep_option = st.selectbox("Duplicate Handling", ["first", "last"])

        if st.button("Remove Duplicates"):
            cleaned_df = remove_duplicates(df.copy(), keep_option)
            st.success("Duplicates Removed")
            st.dataframe(cleaned_df.head())

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:

            selected_col = st.selectbox("Outlier Detection Column", numeric_cols)

            outliers = detect_outliers_iqr(df, selected_col)

            st.metric("Outliers Found", len(outliers))
            st.dataframe(outliers.head())

    # =========================
    # VISUALIZATIONS
    # =========================
    elif page == "Visualizations":

        st.header("📈 Visualizations")

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:

            selected_col = st.selectbox("Numeric Column", numeric_cols)

            st.subheader("Histogram")
            histogram(df, selected_col)

            st.subheader("Box Plot")
            boxplot(df, selected_col)

        if len(numeric_cols) >= 2:

            x_col = st.selectbox("X Axis", numeric_cols, key="x_axis")
            y_col = st.selectbox("Y Axis", numeric_cols, key="y_axis")

            st.subheader("Scatter Plot")
            scatter(df, x_col, y_col)

        st.subheader("Correlation Heatmap")
        correlation_heatmap(df)

        st.subheader("Missing Value Heatmap")
        missing_heatmap(df)

        if st.button("Generate Pair Plot"):
            pair_plot(df)

        categorical_cols = df.select_dtypes(include="object").columns

        if len(categorical_cols) > 0:

            cat_col = st.selectbox("Categorical Column", categorical_cols)

            st.subheader("Count Plot")
            count_plot(df, cat_col)

            st.subheader("Pie Chart")
            pie_chart(df, cat_col)

    # =========================
    # AI INSIGHTS
    # =========================
    elif page == "AI Insights":

        st.header("🧠 AI Insights")

        insights = generate_insights(df)

        for insight in insights:
            st.success(insight)

    # =========================
    # MACHINE LEARNING (FULL UPGRADED)
    # =========================
    elif page == "Machine Learning":

        st.header("🤖 Machine Learning")

        target_column = st.selectbox("Select Target Column", df.columns)

        model_choice = st.selectbox(
            "🧠 Select Model",
            [
                "Auto (All Models)",
                "Logistic Regression",
                "Random Forest",
                "Decision Tree",
                "SVM"
            ]
        )

        if st.button("🚀 Train Models"):

            df_ml = df.copy()
            df_ml = df_ml.dropna()

            # Encode categorical columns
            for col in df_ml.columns:
                if df_ml[col].dtype == "object":
                    df_ml[col] = pd.factorize(df_ml[col])[0]

            X = df_ml.drop(target_column, axis=1)
            y = df_ml[target_column]

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.svm import SVC

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "SVM": SVC()
            }

            results = {}
            trained_models = {}

            st.subheader("📊 Model Results")

            if model_choice == "Auto (All Models)":

                for name, model in models.items():

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    acc = accuracy_score(y_test, preds)

                    results[name] = acc
                    trained_models[name] = model

                    st.success(f"{name}: {acc:.4f}")

                best_model_name = max(results, key=results.get)
                best_model = trained_models[best_model_name]

            else:

                model = models[model_choice]

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)

                results[model_choice] = acc
                best_model = model
                best_model_name = model_choice

                st.success(f"{model_choice}: {acc:.4f}")

            # BEST MODEL
            st.subheader("🏆 Best Model")
            st.info(f"{best_model_name} ({max(results.values()):.4f})")

            # SAVE MODEL
            joblib.dump(best_model, "best_model.pkl")

            with open("best_model.pkl", "rb") as f:
                st.download_button(
                    "📥 Download Model",
                    f,
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )

            # CONFUSION MATRIX
            preds = best_model.predict(X_test)
            cm = confusion_matrix(y_test, preds)

            st.subheader("📊 Confusion Matrix")

            fig, ax = plt.subplots()
            ax.imshow(cm, cmap="Blues")
            st.pyplot(fig)

            # FEATURE IMPORTANCE
            if hasattr(best_model, "feature_importances_"):

                st.subheader("📌 Feature Importance")

                importances = best_model.feature_importances_
                features = X.columns

                fig, ax = plt.subplots()
                ax.barh(features, importances)
                st.pyplot(fig)

            # REPORT
            st.subheader("📄 Classification Report")
            st.text(classification_report(y_test, preds))

else
    st.info("👆 Please upload a CSV file to start")
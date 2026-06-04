import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    r2_score
)

from utils.evaluation import (
    show_confusion_matrix,
    show_classification_report
)

from utils.model_manager import (
    save_model
)


def train_models(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    encoder = LabelEncoder()

    for col in X.columns:

        if X[col].dtype == "object":

            X[col] = encoder.fit_transform(
                X[col].astype(str)
            )

    if y.dtype == "object":

        y = encoder.fit_transform(
            y.astype(str)
        )

    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================
    # Classification
    # =====================

    if len(pd.Series(y).unique()) <= 10:

        st.success(
            "Classification Problem Detected"
        )

        models = {
            "Logistic Regression":
                LogisticRegression(max_iter=1000),

            "Decision Tree":
                DecisionTreeClassifier(),

            "Random Forest":
                RandomForestClassifier(),

            "KNN":
                KNeighborsClassifier()
        }

        results = {}

        best_model = None
        best_score = 0

        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict(
                X_test
            )

            score = accuracy_score(
                y_test,
                preds
            )

            results[name] = score

            if score > best_score:

                best_score = score
                best_model = model

        leaderboard = pd.DataFrame(
            results.items(),
            columns=[
                "Model",
                "Accuracy"
            ]
        ).sort_values(
            by="Accuracy",
            ascending=False
        )

        st.subheader(
            "🏆 Model Leaderboard"
        )

        st.dataframe(
            leaderboard
        )

        best_model_name = max(
            results,
            key=results.get
        )

        st.success(
            f"Best Model: {best_model_name}"
        )

        save_model(
            best_model,
            "models/best_model.pkl"
        )

        st.success(
            "Best model saved successfully!"
        )

        best_predictions = best_model.predict(
            X_test
        )

        st.subheader(
            "📊 Confusion Matrix"
        )

        show_confusion_matrix(
            y_test,
            best_predictions
        )

        st.subheader(
            "📄 Classification Report"
        )

        show_classification_report(
            y_test,
            best_predictions
        )

        if hasattr(
            best_model,
            "feature_importances_"
        ):

            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance":
                best_model.feature_importances_
            })

            st.subheader(
                "🔥 Feature Importance"
            )

            st.bar_chart(
                importance.set_index(
                    "Feature"
                )
            )

    # =====================
    # Regression
    # =====================

    else:

        st.success(
            "Regression Problem Detected"
        )

        models = {
            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(),

            "Random Forest":
                RandomForestRegressor()
        }

        results = {}

        best_model = None
        best_score = -999999

        for name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict(
                X_test
            )

            score = r2_score(
                y_test,
                preds
            )

            results[name] = score

            if score > best_score:

                best_score = score
                best_model = model

        leaderboard = pd.DataFrame(
            results.items(),
            columns=[
                "Model",
                "R2 Score"
            ]
        ).sort_values(
            by="R2 Score",
            ascending=False
        )

        st.subheader(
            "🏆 Model Leaderboard"
        )

        st.dataframe(
            leaderboard
        )

        best_model_name = max(
            results,
            key=results.get
        )

        st.success(
            f"Best Model: {best_model_name}"
        )

        save_model(
            best_model,
            "models/best_model.pkl"
        )

        st.success(
            "Best model saved successfully!"
        )
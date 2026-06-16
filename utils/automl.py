import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Regression Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


def train_automl(df, target, problem_type):

    # -----------------------------
    # Features & Target
    # -----------------------------
    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------
    # Keep Numeric Features
    # -----------------------------
    X = X.select_dtypes(include=["number"])

    # -----------------------------
    # Fill Missing Values
    # -----------------------------
    X = X.fillna(X.mean())

    # -----------------------------
    # Remove Missing Target Rows
    # -----------------------------
    mask = y.notna()

    X = X[mask]
    y = y[mask]

    # -----------------------------
    # Safety Check
    # -----------------------------
    if len(X) == 0:

        return (
            {},
            "No Model",
            0,
            None,
            [],
            {}
        )

    # -----------------------------
    # Train Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = {}

    trained_models = {}

    # ==================================================
    # CLASSIFICATION
    # ==================================================
    if problem_type == "Classification":

        models = {

            "Logistic Regression":
            LogisticRegression(max_iter=1000),

            "Random Forest":
            RandomForestClassifier(
                random_state=42
            ),

            "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

            "KNN":
            KNeighborsClassifier(),

            "SVM":
            SVC(),

            "Naive Bayes":
            GaussianNB()
        }

        for name, model in models.items():

            try:

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

                trained_models[name] = model

            except Exception:
                continue

    # ==================================================
    # REGRESSION
    # ==================================================
    else:

        models = {

            "Linear Regression":
            LinearRegression(),

            "Random Forest":
            RandomForestRegressor(
                random_state=42
            ),

            "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

            "SVR":
            SVR()
        }

        for name, model in models.items():

            try:

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

                trained_models[name] = model

            except Exception:
                continue

    # -----------------------------
    # No Model Trained
    # -----------------------------
    if len(results) == 0:

        return (
            {},
            "No Model",
            0,
            None,
            [],
            {}
        )

    # -----------------------------
    # Best Model
    # -----------------------------
    best_model_name = max(
        results,
        key=results.get
    )

    best_score = results[
        best_model_name
    ]

    best_model = trained_models[
        best_model_name
    ]

    feature_columns = (
        X.columns.tolist()
    )

    # -----------------------------
    # Save Model
    # -----------------------------
    joblib.dump(
        best_model,
        "best_model.pkl"
    )

    # -----------------------------
    # Save Feature Columns
    # -----------------------------
    joblib.dump(
        feature_columns,
        "feature_columns.pkl"
    )

    # -----------------------------
    # Return Results
    # -----------------------------
    return (
        results,
        best_model_name,
        best_score,
        best_model,
        feature_columns,
        trained_models
    )
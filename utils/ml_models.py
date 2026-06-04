import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # AUTO DETECT PROBLEM TYPE
    if y.nunique() <= 10:
        problem = "classification"
    else:
        problem = "regression"

    if problem == "classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC()
        }

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "SVM": SVR()
        }

    results = {}

    best_model = None
    best_name = None
    best_score = -999

    for name, model in models.items():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if problem == "classification":
            score = accuracy_score(y_test, preds.round())
        else:
            score = r2_score(y_test, preds)

        results[name] = score

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    return best_model, best_name, results, X_test, y_test
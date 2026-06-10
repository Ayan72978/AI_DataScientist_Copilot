import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.metrics import (
    accuracy_score,
    r2_score
)


def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if y.nunique() <= 10:
        problem = "classification"
    else:
        problem = "regression"

    if problem == "classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SVM": SVC()
        }

    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "SVR": SVR()
        }

    results = {}

    best_model = None
    best_name = None
    best_score = -999999

    for name, model in models.items():

        try:

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            if problem == "classification":
                score = accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)

            results[name] = round(score, 4)

            try:

                cv_score = cross_val_score(
                    model,
                    X,
                    y,
                    cv=5
                ).mean()

                results[f"{name} CV"] = round(
                    cv_score,
                    4
                )

            except:
                pass

            if score > best_score:

                best_score = score
                best_model = model
                best_name = name

        except Exception as e:

            results[name] = f"Error: {str(e)}"

    return (
        best_model,
        best_name,
        results,
        X_test,
        y_test,
        problem
    )


def predict_single(model, input_data):

    input_array = np.array(
        input_data
    ).reshape(1, -1)

    prediction = model.predict(
        input_array
    )

    return prediction[0]


def save_model(
    model,
    filename="best_model.pkl"
):

    joblib.dump(
        model,
        filename
    )

    return filename


def load_model(
    filename="best_model.pkl"
):

    model = joblib.load(
        filename
    )

    return model

from sklearn.model_selection import cross_val_score

def auto_ml_pipeline(models, X, y):

    results = {}

    best_score = -1
    best_model = None
    best_name = ""

    for name, model in models.items():

        scores = cross_val_score(model, X, y, cv=5)

        avg_score = scores.mean()

        results[name] = avg_score

        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_name = name

    best_model.fit(X, y)

    return best_model, best_name, results


def get_feature_importance(
    model,
    feature_names
):

    try:

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance = (
                model.feature_importances_
            )

            feature_data = list(
                zip(
                    feature_names,
                    importance
                )
            )

            feature_data = sorted(
                feature_data,
                key=lambda x: x[1],
                reverse=True
            )

            return feature_data

        return None

    except Exception:

        return None
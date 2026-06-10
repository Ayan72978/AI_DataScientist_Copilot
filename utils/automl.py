from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def train_automl(df, target, problem_type):

    # -----------------------------
    # Split features and target
    # -----------------------------
    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------
    # Keep numeric only
    # -----------------------------
    X = X.select_dtypes(include=["number"])

    # -----------------------------
    # Handle missing values
    # -----------------------------
    X = X.fillna(X.mean())

    # Remove missing target rows
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Safety check
    if len(X) == 0:
        return {}, "No valid data", {}

    # -----------------------------
    # Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    results = {}
    trained_models = {}

    # =============================
    # CLASSIFICATION
    # =============================
    if problem_type == "Classification":

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            trained_models[name] = model   # ✅ STORE MODEL HERE

            pred = model.predict(X_test)

            results[name] = accuracy_score(y_test, pred)

    # =============================
    # REGRESSION
    # =============================
    else:

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor()
        }

        for name, model in models.items():

            model.fit(X_train, y_train)

            trained_models[name] = model   # ✅ STORE MODEL HERE

            pred = model.predict(X_test)

            results[name] = r2_score(y_test, pred)

    # -----------------------------
    # Best model selection
    # -----------------------------
    best_model = max(results, key=results.get)

    feature_columns = X.columns.tolist()

    return results, best_model, trained_models, feature_columns
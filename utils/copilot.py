import pandas as pd
import numpy as np
import json


# ─────────────────────────────────────────────────────────────────────────────
# CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dataset_summary(df):
    summary = {}
    summary["rows"]           = df.shape[0]
    summary["columns"]        = df.shape[1]
    summary["missing_values"] = int(df.isnull().sum().sum())
    summary["duplicate_rows"] = int(df.duplicated().sum())
    summary["numeric_columns"]     = df.select_dtypes(include=["int64","float64"]).columns.tolist()
    summary["categorical_columns"] = df.select_dtypes(include=["object"]).columns.tolist()
    total_cells = df.shape[0] * df.shape[1]
    summary["quality_score"] = round((1 - summary["missing_values"] / total_cells) * 100, 2) if total_cells > 0 else 0
    return summary


def suggest_target_column(df):
    target = df.columns[-1]
    max_unique = 0
    for col in df.columns:
        if df[col].dtype == "object":
            continue
        u = df[col].nunique()
        if u > max_unique:
            max_unique = u
            target = col
    return target


def detect_problem_type(df, target):
    if df[target].dtype == "object" or df[target].nunique() < 10:
        return "Classification"
    return "Regression"


def suggest_model(df, target):
    if target not in df.columns:
        return "Unknown"
    if df[target].dtype == "object":
        return "Random Forest Classifier" if df[target].nunique() <= 10 else "Logistic Regression"
    return "Random Forest Regressor" if df.shape[0] < 1000 else "Linear Regression"


def data_quality_score(df):
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0
    missing    = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    score = 100
    score -= (missing / total_cells) * 100
    score -= (duplicates / max(len(df), 1)) * 20
    return max(0, round(score))


# ─────────────────────────────────────────────────────────────────────────────
# INSIGHTS & ACTION PLAN
# ─────────────────────────────────────────────────────────────────────────────

def generate_insights(df):
    insights   = []
    missing    = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    insights.append(f"Missing values: {missing}" if missing else "No missing values found.")
    insights.append(f"Duplicate rows: {duplicates}" if duplicates else "No duplicate rows found.")

    if df.shape[0] < 100:
        insights.append("Small dataset — high risk of overfitting.")
    if len(df.select_dtypes(include=["object"]).columns) > 5:
        insights.append("Many categorical columns — consider encoding strategies.")

    numeric_df = df.select_dtypes(include=["int64","float64"])
    if not numeric_df.empty:
        corr      = numeric_df.corr().abs()
        high_corr = [(c1, c2, round(corr.loc[c1,c2],2))
                     for c1 in corr.columns for c2 in corr.columns
                     if c1 < c2 and corr.loc[c1,c2] > 0.85]
        if high_corr:
            pairs = ", ".join([f"{a} & {b} ({v})" for a,b,v in high_corr[:3]])
            insights.append(f"Highly correlated pairs: {pairs}")

        skewed = [f"{col} (skew={round(numeric_df[col].skew(),2)})"
                  for col in numeric_df.columns if abs(numeric_df[col].skew()) > 1]
        if skewed:
            insights.append(f"Skewed columns: {', '.join(skewed[:3])} — log transform may help.")

    return insights


def next_action_plan(df, target):
    actions    = []
    missing    = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    if missing > 0:
        actions.append("Handle missing values (imputation or removal).")
    if duplicates > 0:
        actions.append("Remove duplicate rows.")

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        actions.append(f"Encode categorical columns: {', '.join(cat_cols[:3])}.")

    if len(df.select_dtypes(include=["int64","float64"]).columns) < 2:
        actions.append("Engineer more numeric features.")
    if df.shape[0] < 100:
        actions.append("Collect more data to reduce overfitting risk.")

    if target in df.columns:
        actions.append(f"{detect_problem_type(df, target)} problem — target: '{target}'.")

    actions.append("Perform EDA (distributions, correlations).")
    actions.append("Split data into train/validation/test sets.")
    actions.append("Train and compare multiple ML models.")
    actions.append("Evaluate model performance before deployment.")
    return actions


# ─────────────────────────────────────────────────────────────────────────────
# EDA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def eda_analysis(df):
    """Returns detailed EDA findings as a list of strings."""
    results    = []
    numeric_df = df.select_dtypes(include=["int64","float64"])
    cat_df     = df.select_dtypes(include=["object"])

    results.append(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    results.append(f"**Numeric columns ({len(numeric_df.columns)}):** {', '.join(numeric_df.columns.tolist()) or 'None'}")
    results.append(f"**Categorical columns ({len(cat_df.columns)}):** {', '.join(cat_df.columns.tolist()) or 'None'}")

    # Distributions
    if not numeric_df.empty:
        results.append("--- Numeric Distributions ---")
        for col in numeric_df.columns:
            mn  = round(numeric_df[col].mean(), 3)
            md  = round(numeric_df[col].median(), 3)
            std = round(numeric_df[col].std(), 3)
            mn_val = round(numeric_df[col].min(), 3)
            mx_val = round(numeric_df[col].max(), 3)
            sk  = round(numeric_df[col].skew(), 2)
            results.append(f"  **{col}**: mean={mn}, median={md}, std={std}, min={mn_val}, max={mx_val}, skew={sk}")

    # Correlations
    if len(numeric_df.columns) >= 2:
        results.append("--- Top Correlations ---")
        corr  = numeric_df.corr().abs()
        pairs = [(c1, c2, round(corr.loc[c1,c2],3))
                 for c1 in corr.columns for c2 in corr.columns if c1 < c2]
        pairs.sort(key=lambda x: -x[2])
        for c1, c2, v in pairs[:5]:
            label = "strong" if v > 0.7 else "moderate" if v > 0.4 else "weak"
            results.append(f"  **{c1}** ↔ **{c2}**: {v} ({label})")

    # Outliers (IQR method)
    if not numeric_df.empty:
        results.append("--- Outlier Detection (IQR) ---")
        for col in numeric_df.columns:
            Q1  = numeric_df[col].quantile(0.25)
            Q3  = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((numeric_df[col] < Q1 - 1.5*IQR) | (numeric_df[col] > Q3 + 1.5*IQR)).sum()
            if outliers > 0:
                pct = round(outliers / len(df) * 100, 1)
                results.append(f"  **{col}**: {outliers} outliers ({pct}% of data)")
            else:
                results.append(f"  **{col}**: No outliers detected")

    # Categorical cardinality
    if not cat_df.empty:
        results.append("--- Categorical Columns ---")
        for col in cat_df.columns:
            n_unique = df[col].nunique()
            top_val  = df[col].value_counts().idxmax()
            top_pct  = round(df[col].value_counts().iloc[0] / len(df) * 100, 1)
            results.append(f"  **{col}**: {n_unique} unique values, most common: '{top_val}' ({top_pct}%)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLEANING REPORT
# ─────────────────────────────────────────────────────────────────────────────

def cleaning_report(df):
    """Returns specific cleaning recommendations with Python code."""
    results    = []
    numeric_df = df.select_dtypes(include=["int64","float64"])
    cat_df     = df.select_dtypes(include=["object"])

    # Missing values
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    if not missing_cols.empty:
        results.append("--- Missing Values ---")
        for col, cnt in missing_cols.items():
            pct = round(cnt / len(df) * 100, 1)
            if pct > 50:
                results.append(f"  **{col}**: {cnt} missing ({pct}%) — consider **dropping** this column")
                results.append(f"  ```python\n  df.drop(columns=['{col}'], inplace=True)\n  ```")
            elif col in numeric_df.columns:
                results.append(f"  **{col}**: {cnt} missing ({pct}%) — impute with **median**")
                results.append(f"  ```python\n  df['{col}'].fillna(df['{col}'].median(), inplace=True)\n  ```")
            else:
                results.append(f"  **{col}**: {cnt} missing ({pct}%) — impute with **mode**")
                results.append(f"  ```python\n  df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)\n  ```")
    else:
        results.append("✅ No missing values — no imputation needed.")

    # Duplicates
    dupes = int(df.duplicated().sum())
    if dupes > 0:
        results.append(f"--- Duplicate Rows: {dupes} found ---")
        results.append("  ```python\n  df.drop_duplicates(inplace=True)\n  df.reset_index(drop=True, inplace=True)\n  ```")
    else:
        results.append("✅ No duplicate rows found.")

    # Encoding
    if not cat_df.empty:
        results.append("--- Categorical Encoding ---")
        for col in cat_df.columns:
            n = df[col].nunique()
            if n == 2:
                results.append(f"  **{col}** ({n} values) → **Label Encoding**")
                results.append(f"  ```python\n  from sklearn.preprocessing import LabelEncoder\n  df['{col}'] = LabelEncoder().fit_transform(df['{col}'])\n  ```")
            elif n <= 10:
                results.append(f"  **{col}** ({n} values) → **One-Hot Encoding**")
                results.append(f"  ```python\n  df = pd.get_dummies(df, columns=['{col}'], drop_first=True)\n  ```")
            else:
                results.append(f"  **{col}** ({n} values, high cardinality) → **Target Encoding or drop**")

    # Outliers
    if not numeric_df.empty:
        results.append("--- Outlier Treatment ---")
        for col in numeric_df.columns:
            Q1  = numeric_df[col].quantile(0.25)
            Q3  = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = int(((numeric_df[col] < Q1-1.5*IQR) | (numeric_df[col] > Q3+1.5*IQR)).sum())
            if outliers > 0:
                results.append(f"  **{col}**: {outliers} outliers → cap with IQR method")
                results.append(
                    f"  ```python\n"
                    f"  Q1, Q3 = df['{col}'].quantile([0.25, 0.75])\n"
                    f"  IQR = Q3 - Q1\n"
                    f"  df['{col}'] = df['{col}'].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)\n"
                    f"  ```"
                )

    # Scaling
    if not numeric_df.empty:
        results.append("--- Feature Scaling ---")
        results.append("  Recommended: StandardScaler for most ML models")
        cols_str = str(numeric_df.columns.tolist())
        results.append(
            f"  ```python\n"
            f"  from sklearn.preprocessing import StandardScaler\n"
            f"  scaler = StandardScaler()\n"
            f"  df[{cols_str}] = scaler.fit_transform(df[{cols_str}])\n"
            f"  ```"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ML RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def ml_recommendations(df, target):
    """Returns ML-specific recommendations with code."""
    results    = []
    numeric_df = df.select_dtypes(include=["int64","float64"])

    if target not in df.columns:
        return ["Target column not found in dataset."]

    problem = detect_problem_type(df, target)
    model   = suggest_model(df, target)

    results.append(f"**Problem Type:** {problem}")
    results.append(f"**Target Column:** `{target}`")
    results.append(f"**Recommended Model:** {model}")

    # Feature importance hint
    feature_cols = [c for c in numeric_df.columns if c != target]
    if feature_cols:
        results.append(f"**Suggested Features:** {', '.join(feature_cols)}")

    # Class balance for classification
    if problem == "Classification":
        vc  = df[target].value_counts()
        pct = (vc / len(df) * 100).round(1)
        results.append("--- Class Distribution ---")
        for cls, cnt in vc.items():
            results.append(f"  {cls}: {cnt} ({pct[cls]}%)")
        if pct.min() < 20:
            results.append("⚠️ Imbalanced classes detected — consider SMOTE or class_weight='balanced'")

    # Train/test split code
    features_str = str(feature_cols)
    results.append("--- Train/Test Split ---")
    results.append(
        f"```python\n"
        f"from sklearn.model_selection import train_test_split\n"
        f"X = df[{features_str}]\n"
        f"y = df['{target}']\n"
        f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        f"```"
    )

    # Model training code
    results.append("--- Model Training ---")
    if model == "Random Forest Classifier":
        results.append(
            f"```python\n"
            f"from sklearn.ensemble import RandomForestClassifier\n"
            f"model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
            f"model.fit(X_train, y_train)\n"
            f"print('Accuracy:', model.score(X_test, y_test))\n"
            f"```"
        )
    elif model == "Logistic Regression":
        results.append(
            f"```python\n"
            f"from sklearn.linear_model import LogisticRegression\n"
            f"model = LogisticRegression(max_iter=1000)\n"
            f"model.fit(X_train, y_train)\n"
            f"print('Accuracy:', model.score(X_test, y_test))\n"
            f"```"
        )
    elif model == "Random Forest Regressor":
        results.append(
            f"```python\n"
            f"from sklearn.ensemble import RandomForestRegressor\n"
            f"from sklearn.metrics import mean_squared_error, r2_score\n"
            f"model = RandomForestRegressor(n_estimators=100, random_state=42)\n"
            f"model.fit(X_train, y_train)\n"
            f"preds = model.predict(X_test)\n"
            f"print('R2:', r2_score(y_test, preds))\n"
            f"print('RMSE:', mean_squared_error(y_test, preds, squared=False))\n"
            f"```"
        )
    else:
        results.append(
            "```python\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.metrics import r2_score, mean_squared_error\n"
            "import numpy as np\n"
            f"model = LinearRegression()\n"
            f"model.fit(X_train, y_train)\n"
            f"preds = model.predict(X_test)\n"
            f"print('R2 Score :', round(r2_score(y_test, preds), 4))\n"
            f"print('RMSE     :', round(np.sqrt(mean_squared_error(y_test, preds)), 4))\n"
            f"print('MAE      :', round(np.mean(np.abs(y_test - preds)), 4))\n"
            "```"
        )

    # Feature importance code
    if model in ["Random Forest Classifier", "Random Forest Regressor"]:
        results.append("--- Feature Importance ---")
        results.append(
            f"```python\n"
            f"import pandas as pd\n"
            f"import matplotlib.pyplot as plt\n"
            f"fi = pd.Series(model.feature_importances_, index={features_str})\n"
            f"fi.sort_values().plot(kind='barh')\n"
            f"plt.title('Feature Importance')\n"
            f"plt.tight_layout()\n"
            f"plt.show()\n"
            f"```"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def visualization_recommendations(df, target):
    """Returns chart suggestions with seaborn/matplotlib code."""
    results    = []
    numeric_df = df.select_dtypes(include=["int64","float64"])
    cat_df     = df.select_dtypes(include=["object"])
    problem    = detect_problem_type(df, target) if target in df.columns else "Unknown"

    results.append("--- Recommended Visualizations ---")

    # Histogram for each numeric column
    if not numeric_df.empty:
        cols_str = str(numeric_df.columns.tolist())
        results.append("**1. Distribution plots (histograms)**")
        results.append(
            f"```python\n"
            f"import matplotlib.pyplot as plt\n"
            f"import seaborn as sns\n"
            f"df[{cols_str}].hist(bins=20, figsize=(12,8))\n"
            f"plt.suptitle('Feature Distributions')\n"
            f"plt.tight_layout()\n"
            f"plt.show()\n"
            f"```"
        )

    # Correlation heatmap
    if len(numeric_df.columns) >= 2:
        results.append("**2. Correlation Heatmap**")
        results.append(
            f"```python\n"
            f"import seaborn as sns, matplotlib.pyplot as plt\n"
            f"plt.figure(figsize=(10,6))\n"
            f"sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt='.2f', cmap='coolwarm')\n"
            f"plt.title('Correlation Heatmap')\n"
            f"plt.tight_layout()\n"
            f"plt.show()\n"
            f"```"
        )

    # Target distribution
    if target in df.columns:
        if problem == "Classification":
            results.append(f"**3. Target Class Distribution — `{target}`**")
            results.append(
                f"```python\n"
                f"import seaborn as sns, matplotlib.pyplot as plt\n"
                f"sns.countplot(x='{target}', data=df)\n"
                f"plt.title('Target Class Distribution')\n"
                f"plt.tight_layout()\n"
                f"plt.show()\n"
                f"```"
            )
        else:
            results.append(f"**3. Target Distribution — `{target}`**")
            results.append(
                f"```python\n"
                f"import seaborn as sns, matplotlib.pyplot as plt\n"
                f"sns.histplot(df['{target}'], kde=True)\n"
                f"plt.title('Target Distribution')\n"
                f"plt.tight_layout()\n"
                f"plt.show()\n"
                f"```"
            )

    # Boxplots for outliers
    if not numeric_df.empty:
        cols_str = str(numeric_df.columns.tolist())
        results.append("**4. Boxplots (Outlier Detection)**")
        results.append(
            f"```python\n"
            f"import seaborn as sns, matplotlib.pyplot as plt\n"
            f"fig, axes = plt.subplots(1, {min(len(numeric_df.columns), 4)}, figsize=(14, 4))\n"
            f"for i, col in enumerate({cols_str[:4]}):\n"
            f"    sns.boxplot(y=df[col], ax=axes[i])\n"
            f"    axes[i].set_title(col)\n"
            f"plt.tight_layout()\n"
            f"plt.show()\n"
            f"```"
        )

    # Scatter for top correlated pair
    if len(numeric_df.columns) >= 2:
        corr  = numeric_df.corr().abs()
        pairs = [(c1, c2, corr.loc[c1,c2])
                 for c1 in corr.columns for c2 in corr.columns if c1 < c2]
        if pairs:
            pairs.sort(key=lambda x: -x[2])
            c1, c2, _ = pairs[0]
            hue_part  = f", hue='{target}'" if target in df.columns and detect_problem_type(df, target) == "Classification" else ""
            results.append(f"**5. Scatter Plot — `{c1}` vs `{c2}` (highest correlation)**")
            results.append(
                f"```python\n"
                f"import seaborn as sns, matplotlib.pyplot as plt\n"
                f"sns.scatterplot(x='{c1}', y='{c2}', data=df{hue_part})\n"
                f"plt.title('{c1} vs {c2}')\n"
                f"plt.tight_layout()\n"
                f"plt.show()\n"
                f"```"
            )

    # Categorical plots
    if not cat_df.empty and target in df.columns and target not in cat_df.columns:
        col = cat_df.columns[0]
        results.append(f"**6. Categorical vs Target — `{col}` vs `{target}`**")
        results.append(
            f"```python\n"
            f"import seaborn as sns, matplotlib.pyplot as plt\n"
            f"sns.boxplot(x='{col}', y='{target}', data=df)\n"
            f"plt.title('{col} vs {target}')\n"
            f"plt.xticks(rotation=45)\n"
            f"plt.tight_layout()\n"
            f"plt.show()\n"
            f"```"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SMART CHAT ENGINE — keyword router, no LLM needed
# ─────────────────────────────────────────────────────────────────────────────

def smart_chat(question: str, df: pd.DataFrame) -> list:
    """
    Routes the user's question to the right analysis function.
    Returns a list of answer lines (strings).
    No LLM or API required.
    """
    q      = question.lower().strip()
    target = suggest_target_column(df)

    # ── EDA ──────────────────────────────────────────────────────────────────
    if any(k in q for k in ["eda", "distribution", "correlation", "outlier",
                             "skew", "analyze", "analyse", "statistics",
                             "stats", "describe", "summary", "overview"]):
        return ["## 📊 EDA Analysis"] + eda_analysis(df)

    # ── Cleaning ─────────────────────────────────────────────────────────────
    if any(k in q for k in ["clean", "missing", "null", "duplicate", "encode",
                             "encoding", "scale", "scaling", "impute",
                             "preprocess", "fix", "outlier treatment"]):
        return ["## 🧹 Data Cleaning Report"] + cleaning_report(df)

    # ── ML / Model ───────────────────────────────────────────────────────────
    if any(k in q for k in ["model", "ml", "machine learning", "train", "predict",
                             "classifier", "regressor", "accuracy", "feature importance",
                             "split", "train test", "algorithm", "best model"]):
        return ["## 🤖 ML Recommendations"] + ml_recommendations(df, target)

    # ── Visualization ─────────────────────────────────────────────────────────
    if any(k in q for k in ["visual", "plot", "chart", "graph", "heatmap",
                             "histogram", "scatter", "boxplot", "bar", "seaborn",
                             "matplotlib", "show"]):
        return ["## 📈 Visualization Recommendations"] + visualization_recommendations(df, target)

    # ── Column-specific questions ──────────────────────────────────────────────
    for col in df.columns:
        if col.lower() in q:
            return _column_deep_dive(df, col, target)

    # ── Specific stat questions ────────────────────────────────────────────────
    if any(k in q for k in ["row", "shape", "size", "how many"]):
        return [f"Dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**."]

    if any(k in q for k in ["column", "feature", "variable"]):
        return [f"Columns ({df.shape[1]}): {', '.join(df.columns.tolist())}"]

    if any(k in q for k in ["target", "label", "predict what", "output"]):
        problem = detect_problem_type(df, target)
        return [f"Suggested target column: **{target}** ({problem} problem)"]

    if any(k in q for k in ["quality", "score", "health"]):
        score = data_quality_score(df)
        return [f"Data quality score: **{score}/100**"]

    if any(k in q for k in ["action", "next step", "what should", "recommend",
                              "plan", "todo", "to do", "start"]):
        actions = next_action_plan(df, target)
        return ["## 🚀 Recommended Next Steps"] + [f"**Step {i+1}:** {a}" for i, a in enumerate(actions)]

    if any(k in q for k in ["insight", "finding", "issue", "problem"]):
        insights = generate_insights(df)
        return ["## 💡 Key Insights"] + insights

    if any(k in q for k in ["dtype", "type", "data type"]):
        dtypes = df.dtypes.astype(str).to_dict()
        return ["## Column Data Types"] + [f"  **{c}**: {t}" for c, t in dtypes.items()]

    if any(k in q for k in ["duplicate"]):
        dupes = int(df.duplicated().sum())
        return [f"Duplicate rows: **{dupes}**" + (" — run `df.drop_duplicates()` to remove." if dupes > 0 else " ✅")]

    if any(k in q for k in ["null", "nan", "missing value"]):
        mv = df.isnull().sum()
        mv = mv[mv > 0]
        if mv.empty:
            return ["✅ No missing values in the dataset."]
        return ["## Missing Values"] + [f"  **{c}**: {v}" for c, v in mv.items()]

    # ── Fallback: full overview ────────────────────────────────────────────────
    return _full_overview(df, target)


def _column_deep_dive(df, col, target):
    """Deep analysis of a single column."""
    results = [f"## 🔍 Column Analysis: `{col}`"]
    results.append(f"**Dtype:** {df[col].dtype}")
    results.append(f"**Non-null count:** {df[col].count()} / {len(df)}")
    results.append(f"**Missing:** {df[col].isnull().sum()}")
    results.append(f"**Unique values:** {df[col].nunique()}")

    if df[col].dtype in ["int64", "float64"]:
        results.append(f"**Mean:** {round(df[col].mean(), 3)}")
        results.append(f"**Median:** {round(df[col].median(), 3)}")
        results.append(f"**Std:** {round(df[col].std(), 3)}")
        results.append(f"**Min:** {round(df[col].min(), 3)}")
        results.append(f"**Max:** {round(df[col].max(), 3)}")
        results.append(f"**Skewness:** {round(df[col].skew(), 3)}")
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
        results.append(f"**Outliers (IQR):** {outliers}")
        if col != target and target in df.columns:
            corr = round(df[[col, target]].corr().iloc[0, 1], 3)
            results.append(f"**Correlation with target `{target}`:** {corr}")
    else:
        top5 = df[col].value_counts().head(5).to_dict()
        results.append(f"**Top 5 values:** {top5}")

    return results


def _full_overview(df, target):
    """Full dataset overview when question is unrecognised."""
    summary  = dataset_summary(df)
    problem  = detect_problem_type(df, target)
    model    = suggest_model(df, target)
    score    = data_quality_score(df)
    insights = generate_insights(df)

    results = [
        "## 📋 Dataset Overview",
        f"**Rows:** {summary['rows']}  |  **Columns:** {summary['columns']}",
        f"**Missing values:** {summary['missing_values']}  |  **Duplicates:** {summary['duplicate_rows']}",
        f"**Quality Score:** {score}/100",
        f"**Target column:** {target}  |  **Problem type:** {problem}",
        f"**Suggested model:** {model}",
        "",
        "## 💡 Key Insights",
    ] + insights + [
        "",
        "**💬 Try asking:**",
        "- *Show EDA analysis*",
        "- *What cleaning steps do I need?*",
        "- *Give me ML recommendations*",
        "- *What visualizations should I create?*",
        "- *Tell me about column [column name]*",
    ]
    return results
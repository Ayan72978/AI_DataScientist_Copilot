import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.automl import train_automl
from utils.reports import generate_report

from utils.copilot import (
    dataset_summary,
    suggest_target_column,
    suggest_model,
    generate_insights,
    next_action_plan,
    detect_problem_type,
    data_quality_score
)

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
    pie_chart,
    line_chart,
    distribution_plot,
    violin_plot,
    bar_plot
)

from utils.ml_models import (
    train_models,
    auto_ml_pipeline,
    predict_single,
    get_feature_importance
)

from utils.evaluation import (
    show_confusion_matrix,
    show_classification_report
)
from utils.copilot import (
   dataset_summary, suggest_target_column, detect_problem_type,
   suggest_model, generate_insights, next_action_plan,
   data_quality_score, smart_chat,
   eda_analysis, cleaning_report, ml_recommendations,
   visualization_recommendations,
  )

st.set_page_config(
    page_title="AI Data Scientist Copilot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
page = st.sidebar.radio(
    "🚀 AI Data Scientist Copilot",
    [
        "🏠 Overview",
        "🧹 Data Cleaning",
        "📊 Visualization",
        "🧠 AI Insights",
        "📈 Machine Learning",
        "🤖 Copilot",
        "🚀 Auto ML",          # FIX: was "⚙ Auto ML" in condition below
        "🔮 Predict Data",
        "📄 Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("ML + AI Project Dashboard")

st.title("🤖 AI Data Scientist Copilot")

st.header(page)

st.caption(
    "Upload data, clean it, visualize insights, train ML models and make predictions."
)

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "📁 Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

# -------------------------
# LOAD DATASET
# -------------------------
if uploaded_file is not None:
    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except Exception:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin1")

        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        else:
            st.error("❌ Unsupported file format")
            st.stop()

        st.session_state["df"] = df

        st.success(
            f"✅ Dataset loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns)"
        )

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

# -------------------------
# SAFE DATA ACCESS
# -------------------------
df = st.session_state.get("df", None)

if df is None:
    st.info("📁 Please upload a dataset to continue.")
    st.stop()


# =========================
# 🏠 OVERVIEW
# =========================
if page == "🏠 Overview":

    st.success("🚀 Welcome to AI Data Scientist Copilot")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicates", int(df.duplicated().sum()))

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.divider()

    health = dataset_health(df)

    st.metric("Health Score", f"{health['health_score']}%")
    st.progress(int(health["health_score"]))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Numeric Features",
            len(df.select_dtypes(include="number").columns)
        )

    with col2:
        st.metric(
            "Categorical Features",
            len(df.select_dtypes(include=["object"]).columns)
        )


# =========================
# 🧹 DATA CLEANING
# =========================
elif page == "🧹 Data Cleaning":

    strategy = st.selectbox(
        "Missing Value Strategy",
        ["Mean", "Median", "Mode", "Drop Rows"]
    )

    if st.button("Apply Cleaning"):
        df_cleaned = handle_missing_values(df.copy(), strategy)
        df_cleaned = remove_duplicates(df_cleaned)

        st.session_state["df"] = df_cleaned
        df = df_cleaned

        st.success("✅ Cleaning completed successfully")

    st.dataframe(df.head(), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Cleaned Dataset",
        csv,
        "cleaned_dataset.csv",
        "text/csv"
    )


# =========================
# 📊 VISUALIZATION
# =========================
elif page == "📊 Visualization":

    graph_type = st.selectbox(
        "Select Visualization",
        [
            "Histogram",
            "Box Plot",
            "Scatter Plot",
            "Correlation Heatmap",
            "Missing Values Heatmap",
            "Pair Plot",
            "Count Plot",
            "Pie Chart",
            "Line Chart",
            "Distribution Plot",
            "Violin Plot",
            "Bar Plot"
        ]
    )

    numeric_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    if graph_type == "Histogram" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        histogram(df, col)

    elif graph_type == "Box Plot" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        boxplot(df, col)

    elif graph_type == "Scatter Plot" and len(numeric_cols) >= 2:
        x = st.selectbox("X Axis", numeric_cols)
        y = st.selectbox("Y Axis", numeric_cols)
        scatter(df, x, y)

    elif graph_type == "Correlation Heatmap":
        correlation_heatmap(df)

    elif graph_type == "Missing Values Heatmap":
        missing_heatmap(df)

    elif graph_type == "Pair Plot":
        pair_plot(df)

    elif graph_type == "Count Plot" and cat_cols:
        col = st.selectbox("Column", cat_cols)
        count_plot(df, col)

    elif graph_type == "Pie Chart" and cat_cols:
        col = st.selectbox("Column", cat_cols)
        pie_chart(df, col)

    elif graph_type == "Line Chart" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        line_chart(df, col)

    elif graph_type == "Distribution Plot" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        distribution_plot(df, col)

    elif graph_type == "Violin Plot" and numeric_cols:
        col = st.selectbox("Column", numeric_cols)
        violin_plot(df, col)

    elif graph_type == "Bar Plot" and cat_cols:
        col = st.selectbox("Column", cat_cols)
        bar_plot(df, col)

    else:
        st.warning("⚠ No suitable columns found for this chart type.")


# =========================
# 🧠 AI INSIGHTS
# =========================
elif page == "🧠 AI Insights":

    insights = generate_insights(df)

    for insight in insights:
        st.success(str(insight))

    st.divider()

    st.subheader("Dataset Summary")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"Missing Values: {df.isnull().sum().sum()}")
    st.write(f"Duplicate Rows: {df.duplicated().sum()}")


# =========================
# 📈 MACHINE LEARNING     ← FIX: this entire block was missing
# =========================
elif page == "📈 Machine Learning":

    st.subheader("📈 Machine Learning")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric columns to train a model.")
        st.stop()

    target_col = st.selectbox(
        "🎯 Select Target Column",
        numeric_cols,
        index=len(numeric_cols) - 1
    )

    feature_cols = [c for c in numeric_cols if c != target_col]

    st.write(f"**Features:** {feature_cols}")
    st.write(f"**Target:** {target_col}")

    if st.button("🚀 Train Models"):

        with st.spinner("Training models... please wait"):

            try:
                X = df[feature_cols].fillna(df[feature_cols].mean())
                y = df[target_col].dropna()
                X = X.loc[y.index]

                best_model, best_name, results, X_test, y_test, problem = train_models(X, y)

                # Store in session
                st.session_state["ml_best_model"]   = best_model
                st.session_state["ml_best_name"]     = best_name
                st.session_state["ml_results"]       = results
                st.session_state["ml_X_test"]        = X_test
                st.session_state["ml_y_test"]        = y_test
                st.session_state["ml_problem"]       = problem
                st.session_state["ml_feature_cols"]  = feature_cols

                st.success(f"✅ Training complete! Best Model: **{best_name}**")

            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                st.stop()

    # Show results if already trained
    if "ml_results" in st.session_state:

        st.divider()
        st.subheader("📊 Model Comparison")

        results     = st.session_state["ml_results"]
        best_name   = st.session_state["ml_best_name"]
        best_model  = st.session_state["ml_best_model"]
        X_test      = st.session_state["ml_X_test"]
        y_test      = st.session_state["ml_y_test"]
        problem     = st.session_state["ml_problem"]
        feature_cols = st.session_state["ml_feature_cols"]

        # Filter out CV rows for clean display
        main_results = {k: v for k, v in results.items() if "CV" not in k and isinstance(v, float)}

        results_df = pd.DataFrame(
            list(main_results.items()),
            columns=["Model", "Score"]
        ).sort_values("Score", ascending=False)

        st.dataframe(results_df, use_container_width=True)

        st.success(f"🏆 Best Model: **{best_name}**")

        best_score = main_results.get(best_name, 0)
        metric_label = "Accuracy" if problem == "classification" else "R² Score"
        st.info(f"⭐ {metric_label}: **{best_score:.4f}**")

        # Confusion matrix / classification report for classifiers
        if problem == "classification":
            st.divider()
            st.subheader("📉 Model Evaluation")
            predictions = best_model.predict(X_test)
            show_confusion_matrix(y_test, predictions)
            show_classification_report(y_test, predictions)

        # Feature importance
        st.divider()
        st.subheader("🔍 Feature Importance")

        importance = get_feature_importance(best_model, feature_cols)

        if importance:
            imp_df = pd.DataFrame(importance, columns=["Feature", "Importance"])
            st.bar_chart(imp_df.set_index("Feature"))
        else:
            st.info("Feature importance not available for this model type.")


# =========================
# 🤖 COPILOT
# =========================
elif page == "🤖 Copilot":
 
    summary  = dataset_summary(df)
    target   = suggest_target_column(df)
    problem  = detect_problem_type(df, target)
    score    = data_quality_score(df)
    model    = suggest_model(df, target)
    insights = generate_insights(df)
    actions  = next_action_plan(df, target)
 
    # ── Top metrics ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows",          summary["rows"])
    col2.metric("Columns",       summary["columns"])
    col3.metric("Missing",       summary["missing_values"])
    col4.metric("Quality Score", f"{score}/100")
 
    st.divider()
 
    # ── AI Recommendations ────────────────────────────────────────────────────
    st.subheader("🤖 AI Recommendations")
    r1, r2, r3 = st.columns(3)
    r1.success(f"**Target Column:** {target}")
    r2.info(f"**Suggested Model:** {model}")
    r3.info(f"**Problem Type:** {problem}")
 
    # ── Dataset Readiness ─────────────────────────────────────────────────────
    st.subheader("🎯 Dataset Readiness")
    if df.shape[0] < 100:
        st.warning("⚠️ Small dataset — more data may significantly improve model performance.")
    elif df.shape[0] < 1000:
        st.info("ℹ️ Medium-sized dataset. Suitable for most ML algorithms.")
    else:
        st.success("✅ Large dataset. Excellent for advanced machine learning models.")
 
    # ── Quality Report ────────────────────────────────────────────────────────
    st.subheader("📋 Data Quality Report")
    st.progress(int(score), text=f"Quality Score: {score}/100")
 
    missing    = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
 
    qc1, qc2 = st.columns(2)
    with qc1:
        if missing == 0:
            st.success("✅ No missing values")
        else:
            st.warning(f"⚠️ {missing} missing values detected")
            with st.expander("Missing values by column"):
                mv = df.isnull().sum()
                st.dataframe(
                    mv[mv > 0].reset_index().rename(columns={"index": "Column", 0: "Count"})
                )
    with qc2:
        if duplicates == 0:
            st.success("✅ No duplicate rows")
        else:
            st.warning(f"⚠️ {duplicates} duplicate rows detected")
 
    st.divider()
 
    # ── Four quick-access panels ──────────────────────────────────────────────
    st.subheader("🔍 Quick Analysis Panels")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🧹 Cleaning", "🤖 ML", "📈 Visualizations"])
 
    def render_lines(lines):
        """Render a list of answer lines cleanly with proper code blocks."""
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                # collect full code block
                lang  = line.strip().replace("```", "").strip() or "python"
                code  = []
                i    += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code.append(lines[i])
                    i += 1
                st.code("\n".join(code), language=lang)
            elif line.startswith("✅"):
                st.success(line)
            elif line.startswith("⚠️"):
                st.warning(line)
            elif line.startswith("---"):
                st.markdown(f"**{line}**")
            else:
                st.markdown(line)
            i += 1
 
    with tab1:
        render_lines(eda_analysis(df))
 
    with tab2:
        render_lines(cleaning_report(df))
 
    with tab3:
        render_lines(ml_recommendations(df, target))
 
    with tab4:
        render_lines(visualization_recommendations(df, target))
 
    st.divider()
 
    # ── Smart Insights ────────────────────────────────────────────────────────
    st.subheader("💡 Smart Insights")
    for insight in insights:
        st.success(f"• {insight}")
 
    st.divider()
 
    # ── Action Plan ───────────────────────────────────────────────────────────
    st.subheader("🚀 Action Plan")
    for i, action in enumerate(actions, 1):
        st.warning(f"**Step {i}:** {action}")
 
    st.divider()
 
    # ── Download Reports ──────────────────────────────────────────────────────
    st.subheader("📥 Export Reports")
 
    report_txt = (
        "Dataset Summary\n===============\n"
        f"Rows             : {df.shape[0]}\n"
        f"Columns          : {df.shape[1]}\n"
        f"Missing Values   : {missing}\n"
        f"Duplicate Rows   : {duplicates}\n\n"
        "AI Analysis\n===========\n"
        f"Suggested Target : {target}\n"
        f"Suggested Model  : {model}\n"
        f"Problem Type     : {problem}\n"
        f"Quality Score    : {score}/100\n\n"
        "Key Insights\n============\n"
        + "\n".join(f"- {x}" for x in insights)
        + "\n\nAction Plan\n===========\n"
        + "\n".join(f"- {a}" for a in actions)
    )
 
    report_df = pd.DataFrame({
        "Metric": ["Rows","Columns","Missing Values","Duplicate Rows",
                   "Target Column","Suggested Model","Problem Type","Quality Score"],
        "Value":  [df.shape[0], df.shape[1], missing, duplicates,
                   target, model, problem, score],
    })
 
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            label="📄 Download Text Report",
            data=report_txt,
            file_name="dataset_report.txt",
            mime="text/plain",
        )
    with dl2:
        st.download_button(
            label="📊 Download CSV Report",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="ai_copilot_report.csv",
            mime="text/csv",
        )
 
    st.divider()
 
    # ── Smart Chat ────────────────────────────────────────────────────────────
    st.subheader("💬 Ask Your Dataset")
    st.caption("Ask anything — EDA, cleaning, ML, visualizations, column names, stats. No API needed.")
 
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
 
    # Quick question buttons
    quick_questions = [
        "Show EDA analysis",
        "What cleaning steps do I need?",
        "Give me ML recommendations",
        "What visualizations should I create?",
        "What are the key insights?",
        "What are my next steps?",
        "Show missing values",
        "Show data types",
        "What is the target column?",
        "Show outliers",
        "Show correlations",
        "What is the quality score?",
    ]
 
    st.markdown("**Quick questions:**")
    qb_cols = st.columns(4)
    for idx, qq in enumerate(quick_questions):
        if qb_cols[idx % 4].button(qq, key=f"quick_{idx}"):
            answer_lines = smart_chat(qq, df)
            st.session_state["chat_history"].append({"role": "user",      "content": qq,                        "lines": None})
            st.session_state["chat_history"].append({"role": "assistant", "content": "\n".join(answer_lines),  "lines": answer_lines})
            st.rerun()
 
    st.markdown("---")
 
    # Render existing chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("lines"):
                render_lines(msg["lines"])
            else:
                st.markdown(msg["content"])
 
    # Chat input
    question = st.chat_input("Ask anything about your dataset...")
 
    if question:
        answer_lines = smart_chat(question, df)
        st.session_state["chat_history"].append({"role": "user",      "content": question,                    "lines": None})
        st.session_state["chat_history"].append({"role": "assistant", "content": "\n".join(answer_lines),    "lines": answer_lines})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            render_lines(answer_lines)
 
    # Clear chat
    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear chat history"):
            st.session_state["chat_history"] = []
            st.rerun()




# =========================
# 🚀 AUTO ML              
# =========================
elif page == "🚀 Auto ML":

    st.subheader("🤖 Auto ML — Automatic Model Training")

    target = suggest_target_column(df)

    if target is None:
        st.error("❌ No suitable target column found.")
        st.stop()

    st.write(f"🎯 **Auto-detected Target Column:** `{target}`")

    problem = detect_problem_type(df, target)
    st.write(f"📌 **Problem Type:** `{problem}`")

    # Allow user to override target
    all_cols = df.columns.tolist()
    target_override = st.selectbox(
        "Override Target Column (optional)",
        all_cols,
        index=all_cols.index(target)
    )

    if target_override != target:
        target = target_override
        problem = detect_problem_type(df, target)
        st.info(f"✅ Target overridden to `{target}` | Problem: `{problem}`")

    if st.button("🚀 Train All Models"):

        with st.spinner("Training models... please wait"):

            try:
                # FIX: train_automl returns 6 values — unpack all 6
                results, best_model_name, best_score, best_model, feature_columns, trained_models = train_automl(
                    df, target, problem
                )

                if len(results) == 0:
                    st.error("❌ No model could be trained. Check your dataset.")
                    st.stop()

                # Store everything in session
                st.session_state["best_model"]       = best_model
                st.session_state["best_model_name"]  = best_model_name
                st.session_state["feature_columns"]  = feature_columns
                st.session_state["trained_models"]   = trained_models
                st.session_state["automl_results"]   = results
                st.session_state["automl_score"]     = best_score
                st.session_state["automl_target"]    = target
                st.session_state["automl_problem"]   = problem

            except Exception as e:
                st.error(f"❌ Training error: {e}")
                st.stop()

    # Show results if trained
    if "automl_results" in st.session_state:

        results         = st.session_state["automl_results"]
        best_model_name = st.session_state["best_model_name"]
        best_score      = st.session_state["automl_score"]

        st.divider()
        st.subheader("📊 Model Comparison")

        results_df = pd.DataFrame(
            list(results.items()),
            columns=["Model", "Score"]
        ).sort_values("Score", ascending=False)

        st.dataframe(results_df, use_container_width=True)
        st.bar_chart(results_df.set_index("Model"))

        st.success(f"🏆 Best Model: **{best_model_name}**")

        metric_label = "Accuracy" if st.session_state["automl_problem"] == "Classification" else "R² Score"
        st.info(f"⭐ {metric_label}: **{best_score:.4f}**")

        st.success("✅ Model saved to session — go to 🔮 Predict Data to use it.")


# =========================
# 🔮 PREDICT DATA
# =========================
elif page == "🔮 Predict Data":

    st.subheader("🔮 Predict New Data")

    if "best_model" not in st.session_state:
        st.warning("⚠ Please train a model first in the 🚀 Auto ML page.")
        st.stop()

    if "feature_columns" not in st.session_state:
        st.warning("⚠ Feature columns not found. Please retrain the model.")
        st.stop()

    model        = st.session_state["best_model"]
    feature_cols = st.session_state["feature_columns"]

    if "best_model_name" in st.session_state:
        st.info(f"🏆 Active Model: **{st.session_state['best_model_name']}**")

    st.write(f"📌 Features Required: **{len(feature_cols)}** → {feature_cols}")

    predict_file = st.file_uploader(
        "📁 Upload File for Prediction",
        type=["csv", "xlsx", "xls"]
    )

    if predict_file is not None:

        try:
            fname = predict_file.name.lower()

            if fname.endswith(".csv"):
                try:
                    new_df = pd.read_csv(predict_file, encoding="utf-8")
                except Exception:
                    predict_file.seek(0)
                    new_df = pd.read_csv(predict_file, encoding="latin1")
            elif fname.endswith((".xlsx", ".xls")):
                new_df = pd.read_excel(predict_file)
            else:
                st.error("❌ Unsupported file format")
                st.stop()

            col1, col2 = st.columns(2)
            col1.metric("Rows", new_df.shape[0])
            col2.metric("Columns", new_df.shape[1])

            st.dataframe(new_df.head())

            missing_cols = [c for c in feature_cols if c not in new_df.columns]
            if missing_cols:
                st.warning(f"⚠ Missing feature columns: {missing_cols} — they will be filled with 0.")

            if st.button("🚀 Predict"):

                X_new = new_df.reindex(columns=feature_cols).fillna(0)
                X_new = X_new.select_dtypes(include=["number"])

                predictions = model.predict(X_new)

                result_df = new_df.copy()
                result_df["Prediction"] = predictions

                st.subheader("📊 Prediction Results")
                st.dataframe(result_df)

                st.subheader("📈 Prediction Summary")
                summary_series = result_df["Prediction"].value_counts()
                st.write(summary_series)

                st.subheader("📉 Prediction Distribution")
                st.bar_chart(summary_series)

                st.download_button(
                    label="⬇ Download Predictions",
                    data=result_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

                st.success("✅ Prediction completed successfully!")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")


# =========================
# 📄 REPORTS
# =========================
elif page == "📄 Reports":

    st.subheader("📄 Generate Report")

    missing    = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    target     = suggest_target_column(df)
    model_name = suggest_model(df, target)
    problem    = detect_problem_type(df, target)
    score      = data_quality_score(df)

    report_text = f"""
AI Data Scientist Copilot — Report
=====================================

Dataset Summary
---------------
Rows             : {df.shape[0]}
Columns          : {df.shape[1]}
Missing Values   : {missing}
Duplicate Rows   : {duplicates}

AI Recommendations
------------------
Suggested Target : {target}
Suggested Model  : {model_name}
Problem Type     : {problem}
Quality Score    : {score}/100

Column Info
-----------
{df.dtypes.to_string()}
"""

    st.text_area("Report Preview", report_text, height=300)

    st.download_button(
        label="📥 Download Report (.txt)",
        data=report_text,
        file_name="ai_report.txt",
        mime="text/plain"
    )

    report_df = pd.DataFrame({
        "Metric": [
            "Rows", "Columns", "Missing Values", "Duplicate Rows",
            "Target Column", "Suggested Model", "Problem Type", "Quality Score"
        ],
        "Value": [
            df.shape[0], df.shape[1], missing, duplicates,
            target, model_name, problem, score
        ]
    })

    csv_report = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📊 Download Report (.csv)",
        data=csv_report,
        file_name="ai_report.csv",
        mime="text/csv"
    )

    # PDF Report — only if a model has been trained via Auto ML
    if "best_model_name" in st.session_state and "automl_score" in st.session_state:

        st.divider()
        st.subheader("📑 Generate PDF Report")

        if st.button("📄 Generate PDF"):

            try:
                summary_data = {
                    "rows":    df.shape[0],
                    "columns": df.shape[1]
                }

                pdf_file = generate_report(
                    summary        = summary_data,
                    target         = st.session_state.get("automl_target", target),
                    problem        = st.session_state.get("automl_problem", problem),
                    best_model     = st.session_state["best_model_name"],
                    best_score     = st.session_state["automl_score"],
                    feature_count  = len(st.session_state.get("feature_columns", [])),
                    missing_values = int(missing)
                )

                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="⬇ Download PDF Report",
                        data=f,
                        file_name="ai_report.pdf",
                        mime="application/pdf"
                    )

                st.success("✅ PDF Report generated!")

            except Exception as e:
                st.error(f"❌ PDF generation failed: {e}")
    else:
        st.info("💡 Train a model in 🚀 Auto ML first to enable PDF report generation.")

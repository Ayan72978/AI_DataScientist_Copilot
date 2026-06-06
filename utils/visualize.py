import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def histogram(df, column):

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        df[column].dropna(),
        bins=20
    )

    ax.set_title(f"Histogram - {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    st.pyplot(fig)


def boxplot(df, column):

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(
        y=df[column],
        ax=ax
    )

    ax.set_title(f"Box Plot - {column}")

    st.pyplot(fig)


def scatter(df, x_col, y_col):

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        ax=ax
    )

    ax.set_title(f"{x_col} vs {y_col}")

    st.pyplot(fig)


def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include="number")

    if len(numeric_df.columns) < 2:
        st.warning("Need at least 2 numeric columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)


def missing_heatmap(df):

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.heatmap(
        df.isnull(),
        cbar=False,
        cmap="viridis",
        ax=ax
    )

    ax.set_title("Missing Values Heatmap")

    st.pyplot(fig)


def pair_plot(df):

    numeric_df = df.select_dtypes(include="number")

    if len(numeric_df.columns) < 2:
        st.warning("Need at least 2 numeric columns.")
        return

    pair = sns.pairplot(numeric_df)

    st.pyplot(pair.figure)


def count_plot(df, column):

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=column,
        ax=ax
    )

    plt.xticks(rotation=45)

    ax.set_title(f"Count Plot - {column}")

    st.pyplot(fig)


def pie_chart(df, column):

    counts = df[column].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%"
    )

    ax.set_title(f"Pie Chart - {column}")

    st.pyplot(fig)


def line_chart(df, column):

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df[column].dropna())

    ax.set_title(f"Line Chart - {column}")
    ax.set_xlabel("Index")
    ax.set_ylabel(column)

    st.pyplot(fig)


def distribution_plot(df, column):

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        df[column].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution Plot - {column}")

    st.pyplot(fig)


def violin_plot(df, column):

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.violinplot(
        y=df[column],
        ax=ax
    )

    ax.set_title(f"Violin Plot - {column}")

    st.pyplot(fig)


def bar_plot(df, column):

    counts = df[column].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(8, 4))

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(f"Bar Plot - {column}")

    st.pyplot(fig)
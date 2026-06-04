import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def histogram(df, column):

    fig, ax = plt.subplots()

    ax.hist(
        df[column].dropna(),
        bins=20
    )

    ax.set_title(f"Histogram - {column}")

    st.pyplot(fig)


def boxplot(df, column):

    fig, ax = plt.subplots()

    ax.boxplot(
        df[column].dropna()
    )

    ax.set_title(f"Box Plot - {column}")

    st.pyplot(fig)


def scatter(df, x_col, y_col):

    fig, ax = plt.subplots()

    ax.scatter(
        df[x_col],
        df[y_col]
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    ax.set_title(
        f"{x_col} vs {y_col}"
    )

    st.pyplot(fig)


def correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    if len(numeric_df.columns) < 2:

        st.warning(
            "Need at least 2 numeric columns."
        )
        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title(
        "Correlation Heatmap"
    )

    st.pyplot(fig)


def missing_heatmap(df):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.heatmap(
        df.isnull(),
        cbar=False,
        cmap="viridis",
        ax=ax
    )

    ax.set_title(
        "Missing Values Heatmap"
    )

    st.pyplot(fig)


def pair_plot(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    if len(numeric_df.columns) < 2:

        st.warning(
            "Need at least 2 numeric columns."
        )
        return

    pair = sns.pairplot(
        numeric_df
    )

    st.pyplot(pair.figure)


def count_plot(df, column):

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    sns.countplot(
        x=df[column],
        ax=ax
    )

    plt.xticks(
        rotation=45
    )

    ax.set_title(
        f"Count Plot - {column}"
    )

    st.pyplot(fig)


def pie_chart(df, column):

    fig, ax = plt.subplots()

    df[column].value_counts().plot.pie(
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")

    ax.set_title(
        f"Pie Chart - {column}"
    )

    st.pyplot(fig)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


def histogram(df, column):

    fig = px.histogram(
        df,
        x=column,
        title=f"Histogram - {column}",
        nbins=20
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def boxplot(df, column):

    fig = px.box(
        df,
        y=column,
        title=f"Box Plot - {column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def scatter(df, x_col, y_col):

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        title=f"{x_col} vs {y_col}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    if len(numeric_df.columns) < 2:
        st.warning(
            "Need at least 2 numeric columns."
        )
        return

    corr = numeric_df.corr().round(2)

    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        annotation_text=corr.values.astype(str),
        showscale=True
    )

    fig.update_layout(
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def missing_heatmap(df):

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing": df.isnull().sum().values
    })

    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing",
        title="Missing Values"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def pair_plot(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    if len(numeric_df.columns) < 2:
        st.warning(
            "Need at least 2 numeric columns."
        )
        return

    fig = px.scatter_matrix(
        numeric_df
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def count_plot(df, column):

    counts = (
        df[column]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        column,
        "Count"
    ]

    fig = px.bar(
        counts,
        x=column,
        y="Count",
        title=f"Count Plot - {column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def pie_chart(df, column):

    counts = (
        df[column]
        .value_counts()
        .head(10)
        .reset_index()
    )

    counts.columns = [
        column,
        "Count"
    ]

    fig = px.pie(
        counts,
        names=column,
        values="Count",
        title=f"Pie Chart - {column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def line_chart(df, column):

    fig = px.line(
        df,
        y=column,
        title=f"Line Chart - {column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def distribution_plot(df, column):

    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        title=f"Distribution Plot - {column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def violin_plot(df, column):

    fig = px.violin(
        df,
        y=column,
        box=True,
        title=f"Violin Plot - {column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def bar_plot(df, column):

    counts = (
        df[column]
        .value_counts()
        .head(10)
        .reset_index()
    )

    counts.columns = [
        column,
        "Count"
    ]

    fig = px.bar(
        counts,
        x=column,
        y="Count",
        title=f"Bar Plot - {column}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
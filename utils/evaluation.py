import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

def show_confusion_matrix(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots(
        figsize=(6,4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        "Confusion Matrix"
    )

    st.pyplot(fig)

def show_classification_report(
    y_test,
    predictions
):

    report = classification_report(
        y_test,
        predictions
    )

    st.text(report)
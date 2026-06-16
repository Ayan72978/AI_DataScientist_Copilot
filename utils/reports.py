from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)


def generate_report(
    summary,
    target,
    problem,
    best_model,
    best_score,
    feature_count,
    missing_values
):

    pdf_file = "report.pdf"

    doc = SimpleDocTemplate(
        pdf_file
    )

    styles = getSampleStyleSheet()

    content = []

    # =====================================
    # Title
    # =====================================
    content.append(
        Paragraph(
            "AI Data Scientist Copilot Report",
            styles["Title"]
        )
    )

    content.append(
        Paragraph(
            f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
            styles["Italic"]
        )
    )

    content.append(
        Spacer(1, 20)
    )

    # =====================================
    # Dataset Summary
    # =====================================
    content.append(
        Paragraph(
            "Dataset Summary",
            styles["Heading1"]
        )
    )

    content.append(
        Paragraph(
            f"Total Rows: {summary['rows']}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Total Columns: {summary['columns']}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Missing Values: {missing_values}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Features Used: {feature_count}",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    # =====================================
    # Dataset Health
    # =====================================
    content.append(
        Paragraph(
            "Dataset Health",
            styles["Heading1"]
        )
    )

    total_cells = (
        summary["rows"] *
        summary["columns"]
    )

    missing_percent = 0

    if total_cells > 0:

        missing_percent = (
            missing_values /
            total_cells
        ) * 100

    content.append(
        Paragraph(
            f"Missing Percentage: {missing_percent:.2f}%",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    # =====================================
    # Machine Learning Summary
    # =====================================
    content.append(
        Paragraph(
            "Machine Learning Summary",
            styles["Heading1"]
        )
    )

    content.append(
        Paragraph(
            f"Target Column: {target}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Problem Type: {problem}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Best Model: {best_model}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Best Score: {best_score:.4f}",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    # =====================================
    # Performance Analysis
    # =====================================
    content.append(
        Paragraph(
            "Performance Analysis",
            styles["Heading1"]
        )
    )

    if best_score >= 0.90:

        performance = (
            "Excellent model performance."
        )

    elif best_score >= 0.80:

        performance = (
            "Good model performance."
        )

    elif best_score >= 0.70:

        performance = (
            "Acceptable model performance."
        )

    else:

        performance = (
            "Model may need improvement."
        )

    content.append(
        Paragraph(
            performance,
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    # =====================================
    # Model Evaluation
    # =====================================
    content.append(
        Paragraph(
            "Model Evaluation",
            styles["Heading1"]
        )
    )

    content.append(
        Paragraph(
            f"Selected Model: {best_model}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Evaluation Score: {best_score:.4f}",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    # =====================================
    # Recommendations
    # =====================================
    content.append(
        Paragraph(
            "Recommendations",
            styles["Heading1"]
        )
    )

    recommendations = [
        "Review missing values before training.",
        "Compare multiple models regularly.",
        "Perform feature engineering for better accuracy.",
        "Validate results on unseen datasets.",
        "Monitor model performance over time."
    ]

    for item in recommendations:

        content.append(
            Paragraph(
                f"• {item}",
                styles["BodyText"]
            )
        )

    content.append(
        Spacer(1, 15)
    )

    # =====================================
    # Conclusion
    # =====================================
    content.append(
        Paragraph(
            "Conclusion",
            styles["Heading1"]
        )
    )

    content.append(
        Paragraph(
            "The dataset was analyzed successfully and the best machine learning model was selected automatically. Results should be validated on unseen data before deployment.",
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 20)
    )

    # =====================================
    # Footer
    # =====================================
    content.append(
        Paragraph(
            "Generated by AI Data Scientist Copilot",
            styles["Italic"]
        )
    )

    doc.build(content)

    return pdf_file
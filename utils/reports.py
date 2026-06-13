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
    best_model
):

    pdf_file = "report.pdf"

    doc = SimpleDocTemplate(
        pdf_file
    )

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "AI Data Scientist Copilot Report",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1, 12)
    )

    content.append(
        Paragraph(
            f"Rows: {summary['rows']}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Columns: {summary['columns']}",
            styles["BodyText"]
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

    doc.build(content)

    return pdf_file
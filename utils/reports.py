from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def generate_report(summary, best_model, problem):

    filename = "analysis_report.pdf"

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "AI Data Scientist Copilot Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            str(summary),
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            f"Best Model: {best_model}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Problem Type: {problem}",
            styles["BodyText"]
        )
    )

    doc.build(content)

    return filename
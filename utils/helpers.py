# utils/helpers.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime


def create_pdf(content_text: str, title: str, filename: str) -> str:
    """
    Generate a clean academic-style PDF from text content.
    Used for transcript, summary, and quiz export.
    """
    pdf_path = filename
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, title)

    # Timestamp
    c.setFont("Helvetica", 10)
    c.drawString(
        100,
        730,
        f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M')}"
    )

    # Content
    c.setFont("Helvetica", 12)
    y = 700

    for paragraph in content_text.split("\n"):
        for line in paragraph.split(". "):
            if not line.strip():
                continue

            words = line.split(" ")
            line_buffer = ""

            for word in words:
                if c.stringWidth(line_buffer + " " + word, "Helvetica", 12) < 430:
                    line_buffer = (line_buffer + " " + word).strip()
                else:
                    if y < 100:
                        c.showPage()
                        y = 750
                        c.setFont("Helvetica", 12)
                    c.drawString(100, y, line_buffer)
                    y -= 18
                    line_buffer = word

            if line_buffer:
                if y < 100:
                    c.showPage()
                    y = 750
                    c.setFont("Helvetica", 12)
                c.drawString(100, y, line_buffer)
                y -= 18

        y -= 8

    c.save()
    return pdf_path

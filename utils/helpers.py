# utils/helpers.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os

def create_temp_wav_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def create_pdf(summary_text: str):
    pdf_path = "lecture_notes.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Lecture Notes")
    c.setFont("Helvetica", 12)
    y = 720
    for line in summary_text.split(". "):
        if y < 100:
            c.showPage()
            y = 750
            c.setFont("Helvetica", 12)
        c.drawString(100, y, line.strip())
        y -= 20
    c.save()
    return pdf_path

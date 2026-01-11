import streamlit as st
import tempfile
import torch
import librosa
from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
WHISPER_MODEL = "openai/whisper-tiny"

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎧 LetUNote AI", layout="wide")

# ---------- HEADER ----------
st.markdown("""
<div style="background: linear-gradient(90deg,#00C4FF,#0066FF);
padding:20px;border-radius:12px;text-align:center;color:white;">
<h2>🎧 LetUNote AI</h2>
<p>Audio → Transcript → Notes → Quiz → Flashcards</p>
</div>
""", unsafe_allow_html=True)

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    model.to("cpu")
    return processor, model, summarizer

processor, model, summarizer = load_models()

# ---------- FUNCTIONS ----------
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        path = tmp.name

    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        ids = model.generate(**inputs)

    return processor.batch_decode(ids, skip_special_tokens=True)[0]


def generate_notes(text):
    prompt = f"""
    Convert the following lecture into clear study notes with headings and bullet points:

    {text}
    """
    return summarizer(prompt, max_length=200, min_length=80, do_sample=False)[0]["summary_text"]


def generate_quiz(text):
    prompt = f"""
    Create 5 multiple-choice questions from the lecture.
    Include 4 options and mark the correct answer.

    Lecture:
    {text}
    """
    return summarizer(prompt, max_length=250, min_length=120, do_sample=False)[0]["summary_text"]


def generate_flashcards(text):
    prompt = f"""
    Create flashcards from the lecture.
    Format strictly as:
    Q: Question
    A: Answer

    Lecture:
    {text}
    """
    return summarizer(prompt, max_length=250, min_length=120, do_sample=False)[0]["summary_text"]


def create_pdf(text, title, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, 750, title)
    c.setFont("Helvetica", 12)

    y = 720
    for line in text.split(". "):
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 750
        c.drawString(40, y, line)
        y -= 18

    c.save()
    return filename

# ---------- SESSION STATE ----------
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "notes" not in st.session_state:
    st.session_state.notes = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = ""
if "flashcards" not in st.session_state:
    st.session_state.flashcards = ""

# ---------- UI ----------
st.markdown("### 🎤 Upload Lecture Audio")
audio_file = st.file_uploader(
    "Upload WAV audio file (Streamlit Cloud safe)",
    type=["wav"]
)

if audio_file:
    with st.spinner("Transcribing audio..."):
        st.session_state.transcript = transcribe_audio(audio_file)
        st.session_state.summary = ""
        st.session_state.notes = ""
        st.session_state.quiz = ""
        st.session_state.flashcards = ""

if st.session_state.transcript:
    st.text_area("🗒️ Transcript", st.session_state.transcript, height=260)

    transcript_pdf = create_pdf(
        st.session_state.transcript,
        "Lecture Transcript",
        "lecture_transcript.pdf"
    )

    with open(transcript_pdf, "rb") as f:
        st.download_button("⬇️ Download Transcript PDF", f, "lecture_transcript.pdf")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📝 Study Notes"):
            with st.spinner("Generating notes..."):
                st.session_state.notes = generate_notes(st.session_state.transcript)

    with col2:
        if st.button("❓ Quiz"):
            with st.spinner("Generating quiz..."):
                st.session_state.quiz = generate_quiz(st.session_state.transcript)

    with col3:
        if st.button("🧠 Flashcards"):
            with st.spinner("Generating flashcards..."):
                st.session_state.flashcards = generate_flashcards(st.session_state.transcript)

    with col4:
        if st.button("📚 Summary"):
            with st.spinner("Generating summary..."):
                st.session_state.summary = summarizer(
                    st.session_state.transcript,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )[0]["summary_text"]

# ---------- OUTPUTS ----------
if st.session_state.summary:
    st.text_area("📘 Summary", st.session_state.summary, height=200)

    summary_pdf = create_pdf(
        st.session_state.summary,
        "Lecture Summary",
        "lecture_summary.pdf"
    )

    with open(summary_pdf, "rb") as f:
        st.download_button("⬇️ Download Summary PDF", f, "lecture_summary.pdf")

if st.session_state.notes:
    st.text_area("📝 Study Notes", st.session_state.notes, height=220)

if st.session_state.quiz:
    st.text_area("❓ Quiz", st.session_state.quiz, height=260)

if st.session_state.flashcards:
    st.text_area("🧠 Flashcards", st.session_state.flashcards, height=260)

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>© 2026 Sanjana Krishnan • LetUNote AI</center>",
    unsafe_allow_html=True
)

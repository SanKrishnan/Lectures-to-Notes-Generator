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
<p>Audio → Transcript → Summary</p>
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

    # librosa loads WAV safely without ffmpeg on Streamlit Cloud
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        ids = model.generate(**inputs)

    return processor.batch_decode(ids, skip_special_tokens=True)[0]


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

# ---------- UI ----------
st.markdown("### 🎤 Upload Lecture Audio")
audio_file = st.file_uploader(
    "Upload WAV audio file",
    type=["wav"]   # WAV ONLY for Streamlit Cloud
)

if audio_file:
    with st.spinner("Transcribing audio..."):
        st.session_state.transcript = transcribe_audio(audio_file)
        st.session_state.summary = ""

if st.session_state.transcript:
    st.text_area("🗒️ Transcript", st.session_state.transcript, height=260)

    transcript_pdf = create_pdf(
        st.session_state.transcript,
        "Lecture Transcript",
        "lecture_transcript.pdf"
    )

    with open(transcript_pdf, "rb") as f:
        st.download_button(
            "⬇️ Download Transcript PDF",
            f,
            "lecture_transcript.pdf"
        )

    if st.button("📚 Generate Summary"):
        with st.spinner("Generating summary..."):
            st.session_state.summary = summarizer(
                st.session_state.transcript,
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]

if st.session_state.summary:
    st.text_area("📘 Summary", st.session_state.summary, height=220)

    summary_pdf = create_pdf(
        st.session_state.summary,
        "Lecture Summary",
        "lecture_summary.pdf"
    )

    with open(summary_pdf, "rb") as f:
        st.download_button(
            "⬇️ Download Summary PDF",
            f,
            "lecture_summary.pdf"
        )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>© 2025 Sanjana Krishnan • LetUNote AI</center>",
    unsafe_allow_html=True
)

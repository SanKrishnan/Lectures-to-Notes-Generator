import streamlit as st
import tempfile
import torch
import librosa
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 300
CHUNK_SECONDS = 30
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-tiny")

torch.set_num_threads(2)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎧 LetUNote AI", layout="wide")

# ---------- HEADER ----------
st.markdown("""
<div style="background: linear-gradient(90deg,#00C4FF,#0066FF);
padding:20px;border-radius:12px;text-align:center;color:white;">
<h2>LetUNote AI</h2>
<p>Smart Lecture Assistant for Notes & Summaries</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.header("👤 User Profile")
st.sidebar.text_input("Name")
st.sidebar.text_input("Email")
st.sidebar.selectbox("Role", ["Student", "Teacher", "Other"])
st.sidebar.markdown("---")

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_models():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    asr_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = asr_model.to(device)

    return processor, asr_model, summarizer, device


def get_models():
    if "models" not in st.session_state:
        with st.spinner("Loading AI models (first run only)..."):
            st.session_state.models = load_models()
    return st.session_state.models


# ---------- TRANSCRIPTION ----------
def transcribe_audio(audio_file, processor, asr_model, device):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        path = tmp.name

    speech, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    speech = speech[: SAMPLE_RATE * MAX_AUDIO_SECONDS]

    chunk_size = SAMPLE_RATE * CHUNK_SECONDS
    transcripts = []

    progress = st.progress(0.0)
    total_chunks = max(1, len(speech) // chunk_size)

    for i, start in enumerate(range(0, len(speech), chunk_size)):
        chunk = speech[start:start + chunk_size]

        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            ids = asr_model.generate(
                **inputs,
                task="transcribe",
                max_new_tokens=128
            )

        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        transcripts.append(text)
        progress.progress((i + 1) / total_chunks)

    return " ".join(transcripts)


# ---------- PDF ----------
def create_pdf(text, title, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, title)
    c.setFont("Helvetica", 12)

    y = 720
    for line in text.split(". "):
        if y < 100:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 750
        c.drawString(100, y, line.strip())
        y -= 18

    c.save()
    return filename


# ---------- SESSION STATE ----------
st.session_state.setdefault("transcript", "")
st.session_state.setdefault("summary", "")
st.session_state.setdefault("generate_summary", False)

# ---------- TABS ----------
tab_home, tab_summary = st.tabs(["🏠 Home", "🗒️ Summarize"])

# ---------- HOME ----------
with tab_home:
    uploaded_file = st.file_uploader(
        "🎤 Upload Lecture Audio (.wav / .mp3)",
        type=["wav", "mp3"]
    )

    if uploaded_file:
        processor, asr_model, summarizer, device = get_models()

        with st.spinner("Transcribing audio..."):
            st.session_state.transcript = transcribe_audio(
                uploaded_file, processor, asr_model, device
            )

        # Reset summary when new audio is uploaded
        st.session_state.summary = ""
        st.session_state.generate_summary = False

    if st.session_state.transcript:
        st.text_area(
            "🗒️ Transcript",
            st.session_state.transcript,
            height=260
        )

        pdf = create_pdf(
            st.session_state.transcript,
            "Lecture Transcript",
            "lecture_transcript.pdf"
        )

        with open(pdf, "rb") as f:
            st.download_button(
                "⬇️ Download Transcript (PDF)",
                f,
                "lecture_transcript.pdf"
            )

# ---------- SUMMARY ----------
with tab_summary:
    if not st.session_state.transcript:
        st.info("Upload audio in Home tab first.")
    else:
        processor, asr_model, summarizer, device = get_models()

        if st.button("📚 Generate Summary"):
            with st.spinner("Generating summary..."):
                st.session_state.summary = summarizer(
                    st.session_state.transcript[:2000],
                    max_length=250,
                    min_length=100,
                    do_sample=False
                )[0]["summary_text"]

        if st.session_state.summary:
            st.text_area(
                "📘 Summary",
                st.session_state.summary,
                height=220
            )

            pdf = create_pdf(
                st.session_state.summary,
                "Lecture Summary",
                "lecture_summary.pdf"
            )

            with open(pdf, "rb") as f:
                st.download_button(
                    "⬇️ Download Summary (PDF)",
                    f,
                    "lecture_summary.pdf"
                )
                
# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>© 2025 Sanjana Krishnan • LetUNote AI</center>",
    unsafe_allow_html=True
)

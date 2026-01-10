import streamlit as st
import tempfile
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# ---------------- CONFIG ----------------
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 300
CHUNK_SECONDS = 30
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-tiny")

torch.set_num_threads(2)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎧 LectNotes AI", layout="wide")

# ---------- HEADER ----------
st.markdown("""
<div style="background: linear-gradient(90deg,#00C4FF,#0066FF);
padding:20px;border-radius:12px;text-align:center;color:white;">
<h2>LectNotes AI</h2>
<p>Smart Lecture Assistant for Notes & Summaries</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.header("👤 User Profile")
st.sidebar.text_input("Name")
st.sidebar.text_input("Email")
st.sidebar.selectbox("Role", ["Student", "Teacher", "Other"])
st.sidebar.markdown("---")

language_option = st.sidebar.radio(
    "🌐 Select Output Language",
    ["English", "Hindi"]
)

# ---------- MODEL LOADING ----------
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    asr_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    translator_en_hi = pipeline(
        "translation_en_to_hi",
        model="Helsinki-NLP/opus-mt-en-hi"
    )

    translator_hi_en = pipeline(
        "translation_hi_to_en",
        model="Helsinki-NLP/opus-mt-hi-en"
    )

    return processor, asr_model, summarizer, translator_en_hi, translator_hi_en


processor, asr_model, summarizer, translator_en_hi, translator_hi_en = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model.to(device)

# ---------- HELPERS ----------
def is_hindi(text):
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def get_output_text(text):
    if language_option == "English":
        return translator_hi_en(text)[0]["translation_text"] if is_hindi(text) else text
    else:
        return translator_en_hi(text)[0]["translation_text"] if not is_hindi(text) else text

# ---------- TRANSCRIPTION ----------
def transcribe_audio(audio_file):
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

# ---------- TABS ----------
tabs = st.tabs(["🏠 Home", "🗒️ Summarize"])

# ---------- HOME ----------
with tabs[0]:
    uploaded_file = st.file_uploader(
        "🎤 Upload Lecture Audio (.wav / .mp3)",
        type=["wav", "mp3"]
    )

    if uploaded_file:
        with st.spinner("Transcribing full audio..."):
            st.session_state.transcript = transcribe_audio(uploaded_file)
            st.session_state.summary = ""

    if st.session_state.transcript:
        final_text = get_output_text(st.session_state.transcript)

        st.text_area(
            f"🗒️ Transcript ({language_option})",
            final_text,
            height=260
        )

        pdf = create_pdf(final_text, "Lecture Transcript", "lecture_transcript.pdf")
        with open(pdf, "rb") as f:
            st.download_button("⬇️ Download Transcript", f, "lecture_transcript.pdf")

# ---------- SUMMARY ----------
with tabs[1]:
    if not st.session_state.transcript:
        st.info("Upload audio in Home tab first.")
    else:
        if st.button("📚 Generate Summary"):
            with st.spinner("Summarizing..."):
                st.session_state.summary = summarizer(
                    st.session_state.transcript[:2000],
                    max_length=500,
                    min_length=100,
                    do_sample=False
                )[0]["summary_text"]

        if st.session_state.summary:
            final_summary = get_output_text(st.session_state.summary)

            st.text_area(
                f"📘 Summary ({language_option})",
                final_summary,
                height=220
            )

            pdf = create_pdf(final_summary, "Lecture Summary", "lecture_summary.pdf")
            with open(pdf, "rb") as f:
                st.download_button("⬇️ Download Summary", f, "lecture_summary.pdf")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("<center style='color:gray;'>© 2025 Sanjana Krishnan • LectNotes AI</center>", unsafe_allow_html=True)

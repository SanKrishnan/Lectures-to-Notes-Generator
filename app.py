import streamlit as st
import tempfile
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ---------------- CONFIG ----------------
PORT = int(os.environ.get("PORT", 8501))
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 300     
CHUNK_SECONDS = 30          
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-tiny")

torch.set_num_threads(2)

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎧 LectNotes AI", layout="wide")

# ---------- HEADER ----------
st.markdown("""
<div style="
    background: linear-gradient(90deg,#00C4FF,#0066FF);
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:white;
">
    <h2>LectNotes AI</h2>
    <p style="font-size:16px;">Smart Lecture Assistant for Notes & Summaries</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.header("👤 User Profile")
st.sidebar.text_input("Name")
st.sidebar.text_input("Email")
st.sidebar.selectbox("Role", ["Student", "Teacher", "Other"])
st.sidebar.text_area("Notes / Remarks", height=80)
st.sidebar.markdown("---")

output_mode = st.sidebar.radio(
    "📝 Output Mode",
    ["Original", "Hinglish (Hindi only)"]
)

st.sidebar.info("English → English | Hindi → Hindi / Hinglish")

# ---------- MODEL LOADING ----------
st.info("⏳ Loading AI models... First run may take a few minutes.")

@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    asr_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    return processor, asr_model, summarizer

with st.spinner("🔄 Initializing AI models..."):
    processor, asr_model, summarizer = load_models()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = asr_model.to(device)

# ---------- HELPER FUNCTIONS ----------

def is_hindi(text):
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def hindi_to_hinglish(text):
    try:
        return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except:
        return text

# ---------- TRANSCRIPTION FUNCTION ----------

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(audio_file.read())
        path = tmp.name

    # Load mp3 / wav
    speech, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    # Limit length
    speech = speech[: SAMPLE_RATE * MAX_AUDIO_SECONDS]

    chunk_size = SAMPLE_RATE * CHUNK_SECONDS
    transcripts = []

    progress = st.progress(0.0)
    total_chunks = max(1, len(speech) // chunk_size)

    for i, start in enumerate(range(0, len(speech), chunk_size)):
        chunk = speech[start:start + chunk_size]

        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            ids = asr_model.generate(
                **inputs,
                task="transcribe",
                max_new_tokens=128
            )

        text = processor.batch_decode(
            ids,
            skip_special_tokens=True
        )[0]

        transcripts.append(text)
        progress.progress(min((i + 1) / total_chunks, 1.0))

    return " ".join(transcripts)

# ---------- PDF FUNCTION ----------

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

# ---------- HOME TAB ----------
with tabs[0]:
    st.markdown("### 🎤 Upload Lecture Audio")

    if not st.session_state.transcript:
        uploaded_file = st.file_uploader(
            "Upload audio file (.wav / .mp3)",
            type=["wav", "mp3"]
        )

        if uploaded_file:
            with st.spinner("🎧 Transcribing audio..."):
                st.session_state.transcript = transcribe_audio(uploaded_file)
            st.success("✅ Transcription Completed")
            st.stop()

    if st.session_state.transcript:
        text = st.session_state.transcript

        st.text_area("🗒️ Transcript", text, height=220)

        if output_mode == "Hinglish (Hindi only)" and is_hindi(text):
            st.text_area(
                "🗒️ Transcript (Hinglish)",
                hindi_to_hinglish(text),
                height=220
            )

        pdf = create_pdf(text, "Lecture Transcript", "lecture_transcript.pdf")
        with open(pdf, "rb") as f:
            st.download_button(
                "⬇️ Download Transcript (PDF)",
                f,
                file_name="lecture_transcript.pdf"
            )

        if st.button("🔄 Reset"):
            st.session_state.transcript = ""
            st.session_state.summary = ""

# ---------- SUMMARY TAB ----------
with tabs[1]:
    summary_length = st.slider("Select Summary Length", 50, 300, 150)

    if st.session_state.transcript:
        if st.button("📚 Generate Summary"):
            with st.spinner("Generating summary..."):
                st.session_state.summary = summarizer(
                    st.session_state.transcript[:2000],
                    max_length=summary_length,
                    min_length=summary_length // 2,
                    do_sample=False
                )[0]["summary_text"]

    if st.session_state.summary:
        st.text_area("📘 Summary", st.session_state.summary, height=220)

        pdf = create_pdf(
            st.session_state.summary,
            "Lecture Summary",
            "lecture_summary.pdf"
        )

        with open(pdf, "rb") as f:
            st.download_button(
                "⬇️ Download Summary (PDF)",
                f,
                file_name="lecture_summary.pdf"
            )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>© 2025 Sanjana Krishnan • LectNotes AI</center>",
    unsafe_allow_html=True
)

import os
import re
import streamlit as st
import tempfile
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -------------------- PAGE CONFIG ----------------------------
st.set_page_config(page_title="🎧 LetUNote AI", layout="wide")

# ---------- LOAD MODELS (Whisper + Summarizer + QG) ----------
@st.cache_resource
def load_models():
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        chunk_length_s=30,
        return_timestamps=False
    )

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    qg = pipeline(
        "text2text-generation",
        model="valhalla/t5-small-qg-hl"
    )

    return asr, summarizer, qg


asr_pipe, summarizer_pipe, qg_pipe = load_models()

# ------------------ CLEAN TEXT -------------------------------
def clean_text(text):
    # 1. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 2. Remove non-ASCII glitch characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 3. Sentence segmentation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    cleaned = []
    seen = set()

    for s in sentences:
        s_strip = s.strip()

        # Skip empty or tiny fragments
        if len(s_strip) < 5:
            continue

        # Skip repeated sentences
        s_key = s_strip.lower()
        if s_key in seen:
            continue

        seen.add(s_key)
        cleaned.append(s_strip)

    text = " ".join(cleaned)

    # 4. Remove repeated long phrases 
    text = re.sub(
        r'\b((?:\w+\s+){5,15}\w+)(?:\s+\1)+',
        r'\1',
        text,
        flags=re.IGNORECASE
    )

    # 5. Remove medium-length repeated patterns 
    text = re.sub(
        r'\b((?:\w+\s+){3,8}\w+)(?:\s+\1)+',
        r'\1',
        text,
        flags=re.IGNORECASE
    )

    # 6. Remove repeated short patterns 
    text = re.sub(
        r'\b((?:\w+\s+){1,3}\w+)(?:\s+\1)+',
        r'\1',
        text,
        flags=re.IGNORECASE
    )

    # 7. Remove repeated single words
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

    return text.strip()

# ---------------------- TRANSCRIBE AUDIO ---------------------
def process_audio(audio):
    ext = audio.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(audio.read())
        path = tmp.name

    with st.spinner("🎙️ Transcribing with Whisper Base..."):
        result = asr_pipe(path)

    os.remove(path)

    text = clean_text(result.get("text", "").strip())
    if not text:
        raise RuntimeError("Whisper returned empty transcript.")

    return text

# ------------------------ SUMMARY ----------------------------
def generate_summary(text):

    text = text[:1024]

    with st.spinner("✍️ Generating Summary..."):
        summ = summarizer_pipe(
            text,
            max_length=200,
            min_length=70,
            do_sample=False
        )[0]["summary_text"]

    bullet_summary = "### 📘 Summary\n"
    for line in summ.split(". "):
        if line.strip():
            bullet_summary += f"- {line.strip()}\n"

    return bullet_summary
    
# -------------------- QUESTIONS ------------------------------
def generate_questions(text):

    cut = text[:700]
    highlight = f"highlight: {cut}"

    with st.spinner("❓ Generating Questions..."):
        raw = qg_pipe(highlight)[0]["generated_text"]

    qs = "### ❓ Questions\n"
    for q in raw.split("?"):
        if q.strip():
            qs += f"- {q.strip()}?\n"

    return qs

# ---------------------- PDF CREATION -------------------------
def create_pdf(transcript, summary, questions):
    filename = "LetUNote_Notes.pdf"
    c = canvas.Canvas(filename, pagesize=letter)

    y = 750

    def block(title, body):
        nonlocal y
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, title)
        y -= 20

        c.setFont("Helvetica", 10)
        for line in body.split("\n"):
            if y < 40:
                c.showPage()
                y = 750
            c.drawString(40, y, line[:100])
            y -= 12

        y -= 15

    block("SUMMARY", summary)
    block("QUESTIONS", questions)
    block("TRANSCRIPT", transcript)

    c.save()
    return filename

# --------------------- SESSION STATE -------------------------
for k in ["transcript", "summary", "questions"]:
    if k not in st.session_state:
        st.session_state[k] = ""

# -------------------- HEADER UI ------------------------------
st.markdown("""
<div style="background: linear-gradient(90deg,#00C4FF,#0066FF);
padding:20px;border-radius:12px;text-align:center;color:white;">
<h2>🎧 LetUNote AI</h2>
<p>Your AI-Powered Lecture Notes Generator</p>
</div>
""", unsafe_allow_html=True)

# ------------------------ MAIN UI ----------------------------
col1, col2 = st.columns([1, 1])

# ------------------------- LEFT SIDE -------------------------
with col1:
    st.markdown("### 🎤 Upload Lecture Audio")

    with st.form("audio_form"):
        file = st.file_uploader("Upload WAV/MP3", type=["wav", "mp3"])
        run = st.form_submit_button("🚀 Generate Lecture")

    if run:
        if not file:
            st.error("Please upload a file.")
            st.stop()

        # Transcript
        st.session_state.transcript = process_audio(file)

        # Summary
        st.session_state.summary = generate_summary(
            st.session_state.transcript
        )

        # Questions
        st.session_state.questions = generate_questions(
            st.session_state.transcript
        )

        st.success("🎉 Processing complete!")


# ------------------------RIGHT SIDE -----------------------
with col2:
    if st.session_state.transcript:

        st.markdown("### 📝 Results")
        tab1, tab2, tab3 = st.tabs(["📄 Transcript", "📘 Summary", "❓ Questions"])

        with tab1:
            st.write(st.session_state.transcript)

        with tab2:
            st.markdown(st.session_state.summary)

        with tab3:
            st.markdown(st.session_state.questions)

        # PDF download
        pdf_file = create_pdf(
            st.session_state.transcript,
            st.session_state.summary,
            st.session_state.questions
        )

        with open(pdf_file, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                f,
                file_name="LetUNote_Content.pdf"
            )


# FOOTER -----------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center style='color:#D9DDDC;'>© 2026 Sanjana Krishnan • LetUNote AI</center>", unsafe_allow_html=True)


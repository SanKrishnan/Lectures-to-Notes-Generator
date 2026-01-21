import os
import re
import streamlit as st
import tempfile
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -------------------- PAGE CONFIG ----------------------------
st.set_page_config(
    page_title="LetUNote AI",
    layout="wide",
    # page_icon="🎧"
)

# -------------------- CUSTOM CSS ------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Garamond', serif;
}
/* ===== Background ===== */
.stApp {
    background-color: #F5EDC4;   /* Cream */
}
/* ===== Header ===== */
.header-card {
    background-color: #37627B;   /* Deep teal */
    padding: 40px;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 40px;
    color: #F5EDC4;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
}
/* ===== Main Section Cards ===== */
.section-card {
    background-color: #A35E24;   /* Warm caramel */
    padding: 28px;
    border-radius: 18px;
    color: #F5EDC4;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
}
/* ===== Upload Box ===== */
[data-testid="stFileUploader"] section {
    background-color: #A35E24 !important;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #582417;  /* Coffee brown border */
    color: #F5EDC4 !important;
}
/* ===== Upload Icon/Text ===== */
[data-testid="stFileUploader"] label {
    color: #F5EDC4 !important;
}
/* ===== Buttons ===== */
.stButton button,
.stDownloadButton button {
    background-color: #F4AB2C !important;  /* Golden */
    color: #582417 !important;             /* Text = deep brown */
    border-radius: 12px;
    padding: 10px 28px;
    font-weight: 600;
    border: none;
    font-size: 16px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.25);
}
.stButton button:hover,
.stDownloadButton button:hover {
    background-color: #A35E24 !important;   /* Caramel */
    color: #FFF7E6 !important;              /* Light cream */
}
/* ===== Tabs ===== */
.stTabs [role="tab"] {
    background-color: #F5EDC4 !important;  /* Cream */
    color: #582417 !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 20px;
    font-weight: 600;
    border: 1px solid #A35E24;
}
.stTabs [role="tab"][aria-selected="true"] {
    background-color: #F4AB2C !important;  /* Golden */
    color: #582417 !important;
    border-bottom: none !important;
}
/* ===== Tabs Container Background ===== */
.stTabs [role="tabpanel"] {
    background-color: #A35E24 !important;
    padding: 20px;
    border-radius: 0 10px 10px 10px !important;
    border: 1px solid #582417;
    color: #F5EDC4 !important;
}
/* ===== Text Styling ===== */
h1 {
    color: #F5EDC4 !important;
}
h2, h3, label, p {
    color: #582417 !important;
}
[data-testid="stFileUploader"] div[role="button"],
[data-testid="stFileUploader"] div[role="button"] * {
    color: #FFF7E6 !important;   
}
/* File name specifically */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #F5EDC4 !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] > div:last-child {
    background-color: #37627B !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    margin: 5px;
}
[data-testid="stFileUploader"] > div:last-child * {
    color: #F5EDC4 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        chunk_length_s=30,
        return_timestamps=False
    )

    summarizer = pipeline(
        "summarization", model="facebook/bart-large-cnn"
    )

    qg = pipeline(
        "text2text-generation",
        model="valhalla/t5-small-qg-hl"
    )

    return asr, summarizer, qg


asr_pipe, summarizer_pipe, qg_pipe = load_models()

# ------------------ CLEAN TEXT -------------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    seen = set()

    for s in sentences:
        s_strip = s.strip()
        if len(s_strip) < 5:
            continue
        s_key = s_strip.lower()
        if s_key in seen:
            continue

        seen.add(s_key)
        cleaned.append(s_strip)

    text = " ".join(cleaned)

    text = re.sub(
        r'\b((?:\w+\s+){5,15}\w+)(?:\s+\1)+',
        r'\1', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'\b((?:\w+\s+){3,8}\w+)(?:\s+\1)+',
        r'\1', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'\b((?:\w+\s+){1,3}\w+)(?:\s+\1)+',
        r'\1', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE
    )

    return text.strip()

# ---------------------- TRANSCRIBE AUDIO ---------------------
def process_audio(audio):
    ext = audio.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(audio.read())
        path = tmp.name

    with st.spinner("Transcribing with Whisper Base..."):
        result = asr_pipe(path)

    os.remove(path)

    text = clean_text(result.get("text", "").strip())
    if not text:
        raise RuntimeError("Whisper returned empty transcript.")

    return text

# ------------------------ SUMMARY ----------------------------
def generate_summary(text):
    text = text[:1024]
    with st.spinner("Generating Summary..."):
        summ = summarizer_pipe(
            text,
            max_length=200,
            min_length=70,
            do_sample=False
        )[0]["summary_text"]

    bullet_summary = "### Summary\n"
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
        c.setFont("Times-Bold", 14)
        c.drawString(40, y, title)
        y -= 20
        c.setFont("Times-Roman", 10)

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

# ---------------------- HEADER ------------------------------
st.markdown("""
<div class="header-card">
    <h1> LetUNote AI</h1>
    <h3>Your AI-Powered Lecture Notes Assistant</h3>
</div>
""", unsafe_allow_html=True)


# ------------------------ MAIN LAYOUT ------------------------
col1, col2 = st.columns([2, 3])

# ------------------------- LEFT SIDE -------------------------
with col1:
    st.markdown("### Upload Lecture Audio")

    with st.form("audio_form"):
        file = st.file_uploader(
            "Upload WAV/MP3",
            type=["wav", "mp3"]
        )
        run = st.form_submit_button("Generate Notes")

    if run:
        if not file:
            st.error("Please upload a file.")
            st.stop()

        st.session_state.transcript = process_audio(file)
        st.session_state.summary = generate_summary(st.session_state.transcript)
        st.session_state.questions = generate_questions(st.session_state.transcript)
        st.success("Notes generated successfully!")

    st.markdown('</div>', unsafe_allow_html=True)


# ------------------------ RIGHT SIDE -----------------------
with col2:
    if st.session_state.transcript:
        st.markdown("### Results")

        tab1, tab2, tab3 = st.tabs(["Transcript", "Summary", "Questions"])

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
                "Download PDF",
                f,
                file_name="LetUNote_Content.pdf"
            )
        st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("""<br><center style="color:#37627B; font-weight:600;">
© 2026 Sanjana Krishnan • LetUNote AI</center>
""", unsafe_allow_html=True)

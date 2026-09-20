import os
import re
import streamlit as st
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from faster_whisper import WhisperModel
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
    color: #FFF7E6 !important;
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
def load_asr():
    return WhisperModel("base",device="cpu",compute_type="int8",cpu_threads=2,num_workers=1)
    
@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-6-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# ------------------ CLEAN TEXT -------------------------------
def clean_text(text):
    text = re.sub(r"\s+"," ",text).strip()
    sentences = re.split(r'(?<=[.!?।])\s+',text)
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
    text = re.sub(r'\b((?:\w+\s+){5,15}\w+)(?:\s+\1)+',r'\1',text,flags=re.IGNORECASE)
    text = re.sub(r'\b((?:\w+\s+){3,8}\w+)(?:\s+\1)+',r'\1',text,flags=re.IGNORECASE)
    text = re.sub(r'\b((?:\w+\s+){1,3}\w+)(?:\s+\1)+',r'\1',text,flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)( \1\b)+',r'\1',text,flags=re.IGNORECASE)

    return text.strip()
# ---------------------- TRANSCRIBE AUDIO ---------------------
def process_audio(audio):
    asr_model = load_asr()
    ext = audio.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{ext}"
    ) as tmp:
        tmp.write(audio.read())
        path = tmp.name

    try:
        with st.spinner("Transcribing..."):
            segments, info = asr_model.transcribe(
                path,
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            )
        text = clean_text(text)
        if not text:
            raise RuntimeError("Whisper returned empty transcript.")
        return text
    finally:
        if os.path.exists(path):
            os.remove(path)
# ------------------------ SUMMARY ----------------------------
def generate_summary(text):
    tokenizer, model = load_summarizer()

    sentences = re.split(r'(?<=[.!?।])\s+',text)
    chunks = []

    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) > 900:
            if current:
                chunks.append(current)
            current = sentence
        else:
            current += " " + sentence

    if current:
        chunks.append(current)

    summaries = []

    with st.spinner("Generating English Summary..."):
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )

            summary_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=80,
                min_new_tokens=15,
                num_beams=1,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                do_sample=False
            )

            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            ).strip()

            if summary:
                summaries.append(summary)

    return "### Summary\n" + "\n".join(
        f"- {s}" for s in summaries
    )
# -------------------- QUESTIONS ------------------------------
def generate_questions(text):
    sentences = re.split(r'(?<=[.!?।])\s+',text)
    sentences = [
        s.strip() for s in sentences
        if 10 <= len(s.strip().split()) <= 40
    ]

    if not sentences:
        return "### Questions\n- No clear questions could be generated."

    stopwords = {
        "the","a","an","is","are","was","were","be","been","being",
        "to","of","and","or","in","on","for","with","from","that",
        "this","these","those","we","you","they","it","he","she",
        "will","can","could","would","should","have","has","had",
        "do","does","did","as","at","by","about","into","then",
        "so","if","than","also","just","very","there","here"
    }

    scored = []

    for index,sentence in enumerate(sentences):
        words = re.findall(
            r'[\w\u0900-\u097F]{3,}',
            sentence.lower()
        )

        content_words = [
            w for w in words
            if w not in stopwords
        ]

        if len(content_words) < 4:
            continue

        unique_words = len(set(content_words))
        score = unique_words + min(len(content_words),10) * 0.5

        scored.append((score,index,sentence))

    if not scored:
        return "### Questions\n- No clear questions could be generated."

    # Divide the complete transcript into sections
    selected = []
    total = len(sentences)
    sections = min(5,total)

    for section in range(sections):
        start = section * total // sections
        end = (section + 1) * total // sections

        candidates = [
            item for item in scored
            if start <= item[1] < end
        ]

        if candidates:
            selected.append(max(candidates,key=lambda x:x[0]))

    selected.sort(key=lambda x:x[1])

    questions = []

    for _,_,sentence in selected:
        words = sentence.split()

        # Use the main concept from the sentence
        content = [
            w for w in words
            if w.lower().strip(".,!?()")
            not in stopwords
        ]

        if not content:
            continue

        concept = " ".join(content[:4])

        question = f"What is the role or purpose of {concept}?"

        if question not in questions:
            questions.append(question)

    return "### Questions\n" + "\n".join(
        f"- {q}" for q in questions[:5]
    )
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
    
        try:
            st.session_state.transcript = process_audio(file)
            st.session_state.summary = generate_summary(
                st.session_state.transcript
            )
            st.session_state.questions = generate_questions(
                st.session_state.transcript
            )
            st.success("Notes generated successfully!")
    
        except Exception as e:
            st.error(f"Error while generating notes: {e}")

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
                file_name="Study_Material.pdf"
            )

# FOOTER
st.markdown("""<br><center style="color:#37627B; font-weight:600;">
© 2026 Sanjana Krishnan • LetUNote AI</center>
""", unsafe_allow_html=True)

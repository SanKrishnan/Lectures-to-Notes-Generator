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
    return WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)

@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-6-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# ------------------ CLEAN TEXT -------------------------------
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    cleaned = []
    seen = set()
    for s in sentences:
        s_strip = s.strip()
        if len(s_strip) < 3:continue
        s_key = s_strip.lower()
        if s_key in seen:continue
        seen.add(s_key)
        cleaned.append(s_strip)
    text = " ".join(cleaned)
    text = re.sub(r'\b((?:\w+\s+){5,15}\w+)(?:\s+\1)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b((?:\w+\s+){3,8}\w+)(?:\s+\1)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b((?:\w+\s+){1,3}\w+)(?:\s+\1)+', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
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
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                task="translate"
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
    if not text or not text.strip():
        return "### Summary\n- No summary could be generated."
    tokenizer, model = load_summarizer()
    sentences = re.split(r'(?<=[.!?।])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) > 1500:
            if current:
                chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()

    if current:
        chunks.append(current.strip())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "and", "or", "in", "on", "for", "with", "from", "that",
        "this", "these", "those", "we", "you", "they", "it", "he", "she",
        "will", "can", "could", "would", "should", "have", "has", "had",
        "do", "does", "did", "as", "at", "by", "about", "into", "then",
        "so", "if", "than", "also", "just", "very", "there", "here"
    }

    def get_content_words(s):
        words = re.findall(r'\b[a-zA-Z0-9\u0900-\u097F]{3,}\b', s.lower())
        return [w for w in words if w not in stopwords]

    transcript_words = set(get_content_words(text))

    summary_sentences = []
    seen_summary_lines = set()

    with st.spinner("Generating English Summary..."):
        for chunk in chunks:
            if len(chunk.split()) < 20:
                lines = [chunk]
            else:
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )

                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=90,
                    min_new_tokens=20,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.15,
                    do_sample=False
                )

                summary_out = tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True
                ).strip()
                lines = re.split(r'(?<=[.!?])\s+', summary_out)

            for line in lines:
                line_clean = line.strip()
                if len(line_clean) < 15 or len(line_clean.split()) < 4:
                    continue

                line_content_words = get_content_words(line_clean)
                if not line_content_words:
                    continue

                matched = [w for w in line_content_words if w in transcript_words]
                match_ratio = len(matched) / len(line_content_words)

                if match_ratio >= 0.4:
                    line_key = line_clean.lower()
                    if line_key not in seen_summary_lines:
                        seen_summary_lines.add(line_key)
                        summary_sentences.append(line_clean)

    if not summary_sentences:
        summary_sentences = [s for s in sentences if len(s.split()) >= 6][:5]

    return "### Summary\n" + "\n".join(
        f"- {s}" for s in summary_sentences
    )

# -------------------- QUESTIONS ------------------------------
def generate_questions(text):
    if not text or not text.strip():
        return "### Questions\n- No clear questions could be generated from the transcript."

    sentences = re.split(r'(?<=[.!?।])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]

    if not sentences:
        return "### Questions\n- No clear questions could be generated from the transcript."

    non_concept_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being","to", "of", "and", "or", "in", "on", "for", "with", "from", "that",
        "this", "these", "those", "we", "you", "they", "it", "he", "she","will", "can", "could", "would", "should", "have", "has", "had",
        "do", "does", "did", "as", "at", "by", "about", "into", "then","so", "if", "than", "also", "just", "very", "there", "here", "today",
        "going", "discuss", "look", "first", "next", "finally", "us", "let","when", "using", "like", "which", "what", "where", "how", "who",
        "why", "some", "same", "another", "make", "create", "part", "way","well", "give", "take", "write", "done", "know", "see", "think",
        "study", "studying", "learn", "learning", "understand", "unlike",
        "explore", "exploring", "show", "showing", "contrast", "differ","differs", "different", "normal", "simple", "various", "certain",
        "several", "many", "much", "particular", "main", "key", "important","regulate", "regulates", "regulating", "act", "acts", "acting",
        "work", "works", "working", "use", "uses", "used", "provide","provides", "ensure", "ensures", "allow", "allows", "our", "my", "your",
        "remove", "removing", "removed", "add", "adds", "adding", "added","change", "changes", "changing", "changed", "attach", "attaching",
        "attached", "select", "selects", "selecting", "selected", "safe","save", "saves", "saving", "saved", "refresh", "refreshed", "refreshing"
    }

    def extract_clean_concept(raw_str):
        if not raw_str:
            return ""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', raw_str)
        while words and words[0].lower() in non_concept_words:
            words.pop(0)
        while words and words[-1].lower() in non_concept_words:
            words.pop()
        if not words or len(words) > 3:
            return ""
        return " ".join(words).strip()

    total_sents = len(sentences)
    num_sections = min(5, total_sents)
    section_size = max(1, total_sents // num_sections)

    sections = []
    for i in range(num_sections):
        start = i * section_size
        end = (i + 1) * section_size if i < num_sections - 1 else total_sents
        sec_sents = sentences[start:end]
        if sec_sents:
            sections.append(sec_sents)

    questions = []
    seen_concepts = set()

    for sec in sections:
        q_found = None
        for sentence in sec:
            s_lower = sentence.lower()

            # Pattern A: Difference / Comparison
            m_comp = re.search(r'\b([a-z0-9\s]{3,30}?)\s+(?:differs from|unlike|is different from)\s+([a-z0-9\s]{3,30})', s_lower)
            if m_comp:
                c1 = extract_clean_concept(m_comp.group(1))
                c2 = extract_clean_concept(m_comp.group(2))
                if c1 and c2 and c1.lower() not in seen_concepts and c1.lower() in text.lower():
                    q_found = f"What is the difference between {c1.title()} and {c2.title()} as explained in the lecture?"
                    seen_concepts.add(c1.lower())
                    break

            # Pattern B: Purpose / Function / Mechanism
            m_purp = re.search(r'(?:a|an|the)?\s*([a-z0-9\s]{3,35}?)\s+(?:is used to|helps to|allows us to|allows|functions as|is designed to|is responsible for|works by|enables)\s+([a-z0-9\s]{5,50})', s_lower)
            if m_purp:
                concept = extract_clean_concept(m_purp.group(1))
                if concept and concept.lower() not in seen_concepts:
                    if concept.lower() in text.lower():
                        concept_title = concept.title()
                        q_found = f"How does {concept_title} work or function according to the lecture?"
                        seen_concepts.add(concept.lower())
                        break

            # Pattern C: Definition / Explanation
            m_def = re.search(r'(?:a|an|the)?\s*([a-z0-9\s]{3,35}?)\s+(?:refers to|is defined as|means|is a type of|represents|is characterized by)\s+([a-z0-9\s]{5,50})', s_lower)
            if m_def:
                concept = extract_clean_concept(m_def.group(1))
                if concept and concept.lower() not in seen_concepts:
                    if concept.lower() in text.lower():
                        concept_title = concept.title()
                        q_found = f"How is {concept_title} defined or explained in the lecture?"
                        seen_concepts.add(concept.lower())
                        break

            # Pattern D: Cause / Effect / Importance
            m_cause = re.search(r'(?:a|an|the)?\s*([a-z0-9\s]{3,35}?)\s+(?:causes|leads to|results in|is important for|is critical for)\s+([a-z0-9\s]{5,50})', s_lower)
            if m_cause:
                concept = extract_clean_concept(m_cause.group(1))
                if concept and concept.lower() not in seen_concepts:
                    if concept.lower() in text.lower():
                        concept_title = concept.title()
                        q_found = f"Why is {concept_title} important in the context discussed?"
                        seen_concepts.add(concept.lower())
                        break

        # Fallback for section if no pattern matched: extract clean noun phrase
        if not q_found:
            for sentence in sec:
                words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence)
                content_words = [w for w in words if w.lower() not in non_concept_words]
                if len(content_words) >= 1:
                    concept = extract_clean_concept(" ".join(content_words[:2]))
                    if concept and concept.lower() not in seen_concepts and concept.lower() in text.lower():
                        concept_title = concept.title()
                        q_found = f"What key concept regarding {concept_title} is explained in the lecture?"
                        seen_concepts.add(concept.lower())
                        break

        if q_found:
            questions.append(q_found)

    if not questions:
        return "### Questions\n- No clear questions could be generated from the transcript."

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
            words = line.split(" ")
            line_buf = ""
            for word in words:
                if c.stringWidth(line_buf + " " + word, "Times-Roman", 10) < 520:
                    line_buf = (line_buf + " " + word).strip()
                else:
                    if y < 40:
                        c.showPage()
                        y = 750
                        c.setFont("Times-Roman", 10)
                    c.drawString(40, y, line_buf)
                    y -= 12
                    line_buf = word
            if line_buf:
                if y < 40:
                    c.showPage()
                    y = 750
                    c.setFont("Times-Roman", 10)
                c.drawString(40, y, line_buf)
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

        # Clear previous session data to ensure fresh generation for each upload
        st.session_state.transcript = ""
        st.session_state.summary = ""
        st.session_state.questions = ""

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

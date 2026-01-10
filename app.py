import streamlit as st
import tempfile
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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
user_name = st.sidebar.text_input("Name")
user_email = st.sidebar.text_input("Email")
user_role = st.sidebar.selectbox("Role", ["Student", "Teacher", "Other"])
user_notes = st.sidebar.text_area("Notes / Remarks", height=80)
st.sidebar.markdown("---")

st.sidebar.markdown("### 🌐 Language")
language_option = st.sidebar.radio(
    "Select Output Language",
    ["English", "Hindi"]
)

st.sidebar.info("Welcome to your smart lecture companion ✨")

# ---------- MODEL LOADING MESSAGE ----------
st.info("⏳ Loading AI models... First run may take a few minutes.")

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    translator_hi = pipeline(
        "translation_en_to_hi",
        model="Helsinki-NLP/opus-mt-en-hi"
    )

    return processor, asr_model, summarizer, translator_hi


with st.spinner("🔄 Initializing AI models..."):
    processor, asr_model, summarizer, translator_hi = load_models()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = asr_model.to(device)

# ---------- FUNCTIONS ----------
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(
        speech,
        sampling_rate=16000,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        predicted_ids = asr_model.generate(**inputs)

    return processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]


def translate_to_hindi(text):
    return translator_hi(text)[0]["translation_text"]


def create_pdf(content_text, title, filename):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, title)

    c.setFont("Helvetica", 12)
    y = 720

    for line in content_text.split(". "):
        if y < 100:
            c.showPage()
            y = 750
            c.setFont("Helvetica", 12)

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

    uploaded_file = st.file_uploader(
        "Upload audio file (.wav / .mp3)",
        type=["wav", "mp3"]
    )

    if uploaded_file and not st.session_state.transcript:
        with st.spinner("🎧 Transcribing audio..."):
            st.session_state.transcript = transcribe_audio(uploaded_file)
        st.success("✅ Transcription Completed")

    if st.session_state.transcript:
        with st.expander("🗒️ View Transcript"):
            st.text_area(
                "Transcript",
                st.session_state.transcript,
                height=220
            )

        if language_option == "Hindi":
            with st.expander("🗒️ View Transcript (Hindi)"):
                st.text_area(
                    "Transcript (Hindi)",
                    translate_to_hindi(st.session_state.transcript),
                    height=220
                )

        transcript_pdf = create_pdf(
            st.session_state.transcript,
            "Lecture Transcript",
            "lecture_transcript.pdf"
        )

        with open(transcript_pdf, "rb") as f:
            st.download_button(
                "⬇️ Download Transcript (PDF)",
                data=f,
                file_name="lecture_transcript.pdf",
                mime="application/pdf"
            )

        if st.button("🔄 Reset"):
            st.session_state.transcript = ""
            st.session_state.summary = ""

# ---------- SUMMARY TAB ----------
with tabs[1]:
    summary_length = st.slider(
        "Select Summary Length",
        50, 300, 150
    )

    if st.session_state.transcript:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_summary = st.button("📚 Generate Summary")

        if generate_summary:
            with st.spinner("Generating summary..."):
                st.session_state.summary = summarizer(
                    st.session_state.transcript,
                    max_length=summary_length,
                    min_length=summary_length // 2,
                    do_sample=False
                )[0]["summary_text"]

    if st.session_state.summary:
        st.markdown("""
        <div style="
            background:#f4f8ff;
            padding:15px;
            border-radius:10px;
            border-left:5px solid #0066FF;
        ">
            <h4>📘 Lecture Summary</h4>
        </div>
        """, unsafe_allow_html=True)

        st.text_area(
            "Summary",
            st.session_state.summary,
            height=220
        )

        if language_option == "Hindi":
            st.text_area(
                "Summary (Hindi)",
                translate_to_hindi(st.session_state.summary),
                height=220
            )

        summary_pdf = create_pdf(
            st.session_state.summary,
            "Lecture Summary",
            "lecture_summary.pdf"
        )

        with open(summary_pdf, "rb") as f:
            st.download_button(
                "⬇️ Download Summary (PDF)",
                data=f,
                file_name="lecture_summary.pdf",
                mime="application/pdf"
            )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>© 2025 Sanjana Krishnan • LectNotes AI</center>",
    unsafe_allow_html=True
)

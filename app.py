import streamlit as st
import tempfile
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import google.generativeai as genai

# ---------- CONFIG ----------
st.set_page_config(page_title="🎧 AI Lecture Assistant", layout="wide")
st.markdown("<h2 style='text-align:center;color:#00C4FF;'>AI Lecture Assistant</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Convert Lectures ➜ Summaries ➜ Quizzes ➜ Insights</p>", unsafe_allow_html=True)

# ---------- GEMINI CONFIG ----------
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-pro")

# ---------- SIDEBAR (User Profile) ----------
st.sidebar.image("assets/logo.png", width=120)
st.sidebar.header("👤 User Profile")
user_name = st.sidebar.text_input("Name", placeholder="Enter your full name")
user_email = st.sidebar.text_input("Email", placeholder="Enter your email")
user_role = st.sidebar.selectbox("Role", ["Student", "Teacher", "Other"])
user_notes = st.sidebar.text_area("Notes / Remarks", placeholder="Optional notes...", height=80)
st.sidebar.markdown("---")
st.sidebar.info("Welcome to your smart lecture companion ✨")


# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
/* Gradient top bar */
.stApp > header {
    background: linear-gradient(90deg,#00C4FF,#0066FF);
    color: white;
    height: 60px;
}

/* Big text area for questions */
.big-text {
    height: 150px;
    font-size:16px;
}

/* Glow buttons */
.stButton>button {
    background: linear-gradient(90deg,#00C4FF,#0066FF);
    color:white;
    border-radius:8px;
    height:40px;
    width:100%;
    font-weight:bold;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return processor, asr_model, summarizer

processor, asr_model, summarizer = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_model = asr_model.to(device)

# ---------- FUNCTIONS ----------
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    speech, sr = librosa.load(tmp_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        predicted_ids = asr_model.generate(**inputs)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcript

def summarize_text(text):
    return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

def generate_quiz(context):
    prompt = f"Create 5 multiple-choice quiz questions with options (A-D) from the following lecture:\n\n{context}"
    response = model.generate_content(prompt)
    return response.text

def ask_gemini(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer concisely."
    response = model.generate_content(prompt)
    return response.text

def create_pdf(content_text, title="Lecture Notes", filename="lecture_notes.pdf"):
    """
    Universal PDF generator for Transcript, Summary, and Quiz.
    """
    pdf_path = filename
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, title)
    c.setFont("Helvetica", 12)
    y = 720

    # Split text smartly (handle paragraphs or long sentences)
    for para in content_text.split("\n"):
        for line in para.split(". "):
            if not line.strip():
                continue
            # Wrap text lines that are too long
            words = line.strip().split(" ")
            line_buf = ""
            for w in words:
                if c.stringWidth(line_buf + " " + w, "Helvetica", 12) < 430:
                    line_buf = (line_buf + " " + w).strip()
                else:
                    if y < 100:
                        c.showPage()
                        y = 750
                        c.setFont("Helvetica", 12)
                    c.drawString(100, y, line_buf)
                    y -= 18
                    line_buf = w
            if line_buf:
                if y < 100:
                    c.showPage()
                    y = 750
                    c.setFont("Helvetica", 12)
                c.drawString(100, y, line_buf)
                y -= 18
        y -= 6
        if y < 100:
            c.showPage()
            y = 750
            c.setFont("Helvetica", 12)

    c.save()
    return pdf_path

# ---------- SESSION STATES ----------
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = ""

# ---------- TOP TABS ----------
tabs = st.tabs(["🏠 Home", "🗒️ Summarize", "🧩 Quiz", "💬 Q&A"])

# ---------- HOME TAB ----------
with tabs[0]:
    st.markdown("### 🎤 Upload Lecture Audio (Persistent)")

    if not st.session_state.transcript:
        uploaded_file = st.file_uploader("Upload audio (.wav/.mp3)", type=["wav", "mp3"])
        if uploaded_file:
            with st.spinner("🎧 Transcribing..."):
                st.session_state.transcript = transcribe_audio(uploaded_file)
            st.success("✅ Transcription Complete!")
    else:
        st.info("📝 Transcript already exists. To upload new audio, reset below.")
        uploaded_file = None

    if st.session_state.transcript:
        st.text_area("🗒️ Transcript", st.session_state.transcript, height=250)

        # ✅ Generate transcript PDF
        transcript_pdf = create_pdf(
            st.session_state.transcript,
            title="Lecture Transcript",
            filename="lecture_transcript.pdf"
        )
        with open(transcript_pdf, "rb") as pdf_file:
            st.download_button(
                "⬇️ Download Transcript (PDF)",
                data=pdf_file,
                file_name="lecture_transcript.pdf",
                mime="application/pdf"
            )

        # Reset Button
        if st.button("🔄 Reset Transcript & Upload New Audio"):
            st.session_state.transcript = ""
            st.session_state.summary = ""
            st.session_state.quiz = ""


# ---------- SUMMARIZE TAB ----------
with tabs[1]:
    if st.session_state.transcript:
        if not st.session_state.summary:
            if st.button("📚 Generate Summary"):
                with st.spinner("Summarizing..."):
                    st.session_state.summary = summarize_text(st.session_state.transcript)

        if st.session_state.summary:
            st.success("✅ Summary Ready!")
            st.text_area("📘 Summary", st.session_state.summary, height=250)

            # ✅ Generate Summary PDF
            summary_pdf = create_pdf(
                st.session_state.summary,
                title="Lecture Summary",
                filename="lecture_summary.pdf"
            )
            with open(summary_pdf, "rb") as pdf_file:
                st.download_button(
                    "⬇️ Download Summary (PDF)",
                    data=pdf_file,
                    file_name="lecture_summary.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("Upload and transcribe audio first from Home tab.")


# ---------- QUIZ TAB ----------
with tabs[2]:
    if st.session_state.transcript:
        if not st.session_state.quiz:
            if st.button("🎯 Generate Quiz"):
                with st.spinner("Generating quiz..."):
                    st.session_state.quiz = generate_quiz(st.session_state.transcript)

        if st.session_state.quiz:
            st.success("✅ Quiz Ready!")
            st.text_area("📝 Quiz Questions", st.session_state.quiz, height=300)

            # ✅ Generate Quiz PDF
            quiz_pdf = create_pdf(
                st.session_state.quiz,
                title="Lecture Quiz",
                filename="lecture_quiz.pdf"
            )
            with open(quiz_pdf, "rb") as pdf_file:
                st.download_button(
                    "⬇️ Download Quiz (PDF)",
                    data=pdf_file,
                    file_name="lecture_quiz.pdf",
                    mime="application/pdf"
                )
    else:
        st.warning("Upload and transcribe audio first from Home tab.")

# ---------- Q&A TAB ----------
with tabs[3]:
    if st.session_state.transcript:
        question = st.text_area("💬 Ask a question about the lecture:", height=150, key="qna_box")
        if st.button("Get Answer"):
            if question.strip():
                with st.spinner("Thinking..."):
                    answer = ask_gemini(question, st.session_state.transcript)
                st.success("✅ Answer:")
                st.text_area("Answer", answer, height=150)
            else:
                st.warning("Please type a question.")
    else:
        st.warning("Upload and transcribe audio first from Home tab.")

st.markdown("---")
st.caption("© 2025 AyushKumar Sharma | Design by AyushKumar Sharma")
st.caption("⚡ Built with Streamlit + Whisper + BART + Gemini AI")

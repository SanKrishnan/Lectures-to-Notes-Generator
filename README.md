---
title: LetUNote AI
emoji: 🎧
colorFrom: blue
colorTo: yellow
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 🎧 LetUNote AI

An AI-powered lecture assistant that converts audio into study-ready notes, summaries & questions

LetUNote AI is a web-based application that helps students and educators transform lecture recordings into clean transcripts, structured summaries, and exam-style questions — all exportable as a PDF. Powered by modern Speech-to-Text and NLP models, it makes studying easier and faster.

---

🌐 **Live Demo:**  
👉 https://huggingface.co/spaces/SanKrishnan/LetUNote_AI

## 🚀 Overview

LetUNote AI streamlines the note-taking process by automatically generating learning material from lecture audio.
Built with Streamlit and Hugging Face Transformers, the app converts spoken lectures into:

- Cleaned transcript (no repetitions or glitches)

- Concise AI-generated summary

- Structured exam-style questions

- Downloadable formatted PDF

LetUNote AI enables students to focus on learning instead of manually writing notes.
---

## ✨ Key Features

- 🎤 Audio Transcription
Converts WAV/MP3 lecture audio into readable text using OpenAI Whisper.

- 📘 AI-Powered Summarization
Generates clean, concise summaries using BART (facebook/bart-large-cnn).

- ❓ Automatic Question Generation
Creates exam-style MCQ/short-answer style questions using T5 (valhalla/t5-small-qg-hl).

= 📄 PDF Export
Allows users to download transcript, summary, and questions in a polished PDF layout.

- 🎨 Clean Streamlit UI
Academic-friendly, responsive interface with tabs for Transcript, Summary, and Questions.

---

## 🛠️ Tech Stack

| Category         | Technology                    |
| ---------------- | ----------------------------- |
| Frontend         | Streamlit                     |
| Speech-to-Text   | OpenAI Whisper (Hugging Face) |
| Summarization    | facebook/bart-large-cnn       |
| Question Gen     | t5-small-qg-hl                |
| Audio Processing | Librosa, SoundFile            |
| PDF Generation   | ReportLab                     |
| Backend          | Python                        |

---

## 📂 Project Structure
```bash
LetUNote_AI/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── README.md           # Documentation
└── .streamlit/         # Streamlit configuration (optional)

```
---

## ⚡ Installation & Usage

### Prerequisites
- Python 3.10+
- Git

### Steps

```bash
git clone https://github.com/SanKrishnan/LetUNote_AI.git
cd LetUNote_AI
pip install -r requirements.txt
streamlit run app.py
```
## 📋 How It Works
- Upload a lecture audio file (.wav or .mp3)

- Whisper transcribes the speech into clean English text

- The transcript is processed to remove repetition/hallucination

- A summary is generated using BART

- Questions are generated using T5

- Output is displayed in Transcript / Summary / Questions tabs

- User can download the entire content as a PDF
- 
## 🌐 Deployment
LetUNote AI is optimized for free-tier deployment and runs efficiently on:

- Hugging Face Spaces (CPU)

- Local machines (Windows/Mac/Linux)

No GPU required.

## 🎓 Use Cases
- Automated lecture note creation

- Exam preparation

- Fast revision tool

- Accessible learning for audio-based students

- Creating study material from seminars & workshops

## ⚠️ Known Limitations
- Best performance with clear English audio

- Long/noisy audio may reduce accuracy

- CPU processing may take longer for long lectures

## 🔮 Future Enhancements
- Multilingual transcription and translation

- Timestamped transcripts

- Keyword extraction and topic highlighting

- Enhanced quiz difficulty levels

- Integration with Notion / Anki


## 👩‍💻 Author
Sanjana Krishnan


🔗 GitHub: https://github.com/SanKrishnan

⭐ If you find this project helpful, consider giving it a star!

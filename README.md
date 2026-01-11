# 🎧 LetUNote AI  
**An AI-powered lecture assistant that converts audio into study-ready notes and summaries**

LetUNote AI is a web-based application that helps students and educators transform lecture recordings into accurate transcripts, concise summaries, and downloadable PDFs using modern Speech-to-Text and Natural Language Processing models.

---

🌐 **Live Demo:**  
👉 https://huggingface.co/spaces/SanKrishnan/LetUNote_AI

## 🚀 Overview

LetUNote AI streamlines the learning process by automating note creation from lecture audio. Built with Streamlit and Hugging Face Transformers, the application converts spoken lectures into lecture transcripts, concise summaries, structured study notes, quiz questions, and flashcards, enabling students to focus on understanding concepts rather than manually taking notes.
---

## ✨ Key Features

- 🎤 Audio Transcription
Converts English lecture audio files (.wav) into accurate text using OpenAI Whisper.

- 📘 AI-Powered Summarization
Generates concise, context-aware summaries from long lecture transcripts using transformer-based NLP models.

- 📝 Study Notes, Quiz & Flashcards
Automatically creates structured study notes, quiz questions, and flashcards from lecture content using generative AI.

- 📄 PDF Export
Allows users to download transcripts and summaries as clean, formatted PDF documents.

- 🎨 Interactive UI
Academic-friendly interface built with Streamlit for smooth and intuitive interaction.

---

## 🛠️ Tech Stack

| Category         | Technology                |
| ---------------- | ------------------------- |
| Frontend         | Streamlit                 |
| Speech-to-Text   | OpenAI Whisper            |
| Text Generation  | DistilBART (Hugging Face) |
| Audio Processing | Librosa                   |
| PDF Generation   | ReportLab                 |
| Backend          | Python                    |


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

- The audio is transcribed into English text using Whisper

- Generate an AI-powered summary of the transcript

- Optionally translate the output into Hindi

- Download the transcript or summary as a PDF

## 🌐 Deployment
LetUNote AI is optimized for free-tier cloud deployment and runs successfully on Hugging Face Spaces (CPU)

## 🎓 Use Cases
- Automated lecture note generation

- Quick exam revision

- Audio-based learning accessibility

- AI-assisted study material creation

## ⚠️ Known Limitations
- Best performance with clear English audio

- Long audio files (>5 minutes) may take longer to process on free-tier CPU

- Currently supports WAV audio only for cloud compatibility

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

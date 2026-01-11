# 🎧 LetUNote AI  
**An AI-powered lecture assistant that converts audio into study-ready notes and summaries**

LetUNote AI is a web-based application that helps students and educators transform lecture recordings into accurate transcripts, concise summaries, and downloadable PDFs using modern Speech-to-Text and Natural Language Processing models.

---

🌐 **Live Demo:**  
👉 https://huggingface.co/spaces/SanKrishnan/LetUNote_AI

## 🚀 Overview

LetUNote AI streamlines the learning process by automating note creation from lecture audio. Built with Streamlit and Hugging Face transformers, the application provides a simple, interactive interface for converting audio content into structured textual material suitable for revision and study.

---

## ✨ Key Features

- 🎤 **Audio Transcription**  
  Converts English lecture audio files (`.wav`, `.mp3`) into accurate text using OpenAI Whisper.

- 📘 **AI-Powered Summarization**  
  Generates concise and context-aware summaries from long transcripts using transformer-based NLP models.

- 🌐 **English → Hindi Translation**  
  Provides optional translation of transcripts and summaries into Hindi for multilingual learners.

- 📄 **PDF Export**  
  Allows users to download transcripts and summaries as clean, formatted PDF documents.

- 🎨 **Interactive UI**  
  Clean, academic-friendly interface built with Streamlit for easy interaction.

---

## 🛠️ Tech Stack

| Category | Technology |
|--------|------------|
| Frontend | Streamlit |
| Speech-to-Text | OpenAI Whisper |
| Summarization | DistilBART (Hugging Face) |
| Translation | Helsinki NLP (Opus-MT) |
| Audio Processing | Librosa |
| PDF Generation | ReportLab |
| Backend | Python |

---

## 📂 Project Structure
```bash
LetUNote-AI/
│
├── app.py # Main Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Documentation
└── .streamlit/ # Streamlit configuration
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

- Multilingual academic content creation

## ⚠️ Known Limitations
-cBest performance with clear English audio

- Long audio files (>5 minutes) may take longer to process on free-tier CPU

- Hindi audio transcription is not supported in the free version

## 🔮 Future Enhancements
- Quiz and question generation from lecture content

- Support for additional Indian languages

- Timestamped transcripts

- Keyword extraction and topic highlighting

## 👩‍💻 Author
Sanjana Krishnan


🔗 GitHub: https://github.com/SanKrishnan

⭐ If you find this project helpful, consider giving it a star!

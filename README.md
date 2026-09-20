# 🎧 LetUNote AI

### AI-Powered Lecture Notes Assistant

LetUNote AI converts lecture audio into structured study material by generating:

- 🎙️ English transcripts
- 📘 Concise lecture summaries
- ❓ Transcript-grounded study questions
- 📄 Downloadable PDF notes

It is designed to work with lectures across different subjects, including computer science, mathematics, science, business, and other academic topics.

## 🌐 Live Demo

**Hugging Face Space:**  
https://huggingface.co/spaces/SanKrishnan/LetUNote_AI

**GitHub Repository:**  
https://github.com/SanKrishnan/Lectures-to-Notes-Generator

---

## 🚀 Overview

LetUNote AI allows users to upload a lecture recording in WAV or MP3 format and automatically transform it into study-ready material.

The application processes the **complete uploaded audio** and presents the results in three sections:

**Transcript**  
The complete lecture is transcribed and translated into English.

**Summary**  
The transcript is processed across the lecture to produce concise, grounded summary points.

**Questions**  
Study questions are generated from concepts and explanations present in the transcript rather than from a fixed subject-specific question set.

The generated transcript, summary, and questions can also be exported together as a PDF.

---

## ✨ Key Features

### 🎤 Lecture Transcription
- Supports WAV and MP3 audio
- Processes the complete uploaded recording
- Uses Faster-Whisper for speech recognition
- Supports multilingual speech with English translation
- Includes repetition cleanup

### 📘 Lecture Summarization
- Uses the complete transcript
- Processes longer lectures in chronological sections
- Generates concise English summary points
- Keeps summary content grounded in the lecture transcript

### ❓ Study Question Generation
- Uses the lecture transcript directly
- Generates questions from concepts, explanations, relationships, comparisons, and other meaningful information present in the lecture
- Designed to work across different academic subjects
- Avoids hardcoded subject-specific questions

### 📄 PDF Export
Downloads the transcript, summary, and generated questions as a formatted PDF.

### 🎨 Streamlit Interface
- Simple academic-focused interface
- Transcript, Summary, and Questions tabs
- Upload and generate workflow
- PDF download option

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Speech-to-Text | Faster-Whisper |
| Speech Model | Whisper Base |
| Summarization | `sshleifer/distilbart-cnn-6-6` |
| Question Generation | Transcript-grounded rule-based extraction |
| PDF Generation | ReportLab |
| Programming Language | Python |
| Deployment | Hugging Face Spaces |

---

## 🔄 How It Works

```text
Lecture Audio
     ↓
Faster-Whisper
     ↓
Complete English Transcript
     ↓
 ┌───────────────┬────────────────┐
 ↓               ↓                ↓
Summary       Questions        Transcript
 ↓               ↓                ↓
        Study Material
              ↓
          PDF Export
```
## Processing Flow
- Upload a WAV or MP3 lecture.
- Faster-Whisper processes the complete recording.
- Non-English speech can be translated into English.
- The transcript is cleaned to reduce repeated text.
- The complete transcript is divided into sections for summarization.
- Summary points are checked against transcript content.
- Meaningful questions are extracted from concepts and explanations in the transcript.
- The transcript, summary, and questions are displayed in Streamlit.
- The complete study material can be downloaded as a PDF.


## 📂 Project Structure
```text
Lectures-to-Notes-Generator/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── .streamlit/
│   └── config.toml
│
└── examples/
    ├── Study_Material.pdf
    └── Write it Right_Tricks of the Trade.mp3
```
The examples/ directory contains a sample lecture input and its generated PDF output.

## ⚡ Run Locally
# Prerequisites
- Python 3.10+
- Git

## Clone the repository
- git clone https://github.com/SanKrishnan/Lectures-to-Notes-Generator.git
- cd Lectures-to-Notes-Generator
- Install dependencies
- pip install -r requirements.txt
- Start the application
- streamlit run app.py

The application will open in your browser through Streamlit.

## ☁️ Deployment

LetUNote AI is designed for deployment on Hugging Face Spaces using Streamlit.

The application is designed to run on CPU and downloads the required models at runtime.

Hugging Face Space

https://huggingface.co/spaces/SanKrishnan/LetUNote_AI

## 🎓 Use Cases
- Lecture note generation
- Exam preparation
- Revision material creation
- Seminar and workshop transcription
- Study material generation from recorded lectures
- Converting long lecture recordings into structured notes

## ⚠️ Limitations
- Transcription quality depends on audio clarity.
- Heavy background noise can reduce transcription accuracy.
- CPU inference can take longer for long recordings.
- Speech recognition may occasionally contain transcription errors.
- Generated summaries and questions are grounded in the transcript, but model-generated content can still require human review.

## 🔮 Future Enhancements
- Timestamped transcripts
- Improved multilingual support
- Topic and keyword extraction
- Adjustable summary length
- Multiple question difficulty levels
- MCQ generation
- Notion and Anki integration

## 👩‍💻 Author
Sanjana Krishnan

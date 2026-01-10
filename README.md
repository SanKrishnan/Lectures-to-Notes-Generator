# 🎧 LectNotes AI  
### Smart Lecture Assistant for Audio-to-Notes & Summaries

**LectNotes AI** is an AI-powered web application that converts lecture audio into structured transcripts, concise summaries, and multilingual outputs. It is designed for students, educators, and multilingual learners who want to automate lecture note creation efficiently.

---

## ✨ Key Features

### 🎤 Speech-to-Text Conversion  
Converts lecture audio files (**MP3, WAV, M4A**) into accurate transcripts using **OpenAI Whisper**.

### 📘 AI-Powered Summarization  
Generates concise and meaningful summaries from long transcripts using the **DistilBART** transformer model.

### 🌐 Multilingual Support (English & Hindi)  
- English audio → English text  
- Hindi audio → Hindi text  
- Optional English ↔ Hindi translation

### 📄 PDF Export  
Download transcripts and summaries as professionally formatted PDF files.

### 🎨 Interactive UI  
Clean, academic-friendly interface built using **Streamlit**.

---

## 🛠️ Tech Stack

| Category | Technology |
|--------|-----------|
| Framework | Streamlit |
| Speech-to-Text | OpenAI Whisper |
| Summarization | DistilBART (Hugging Face) |
| Translation | Helsinki NLP Opus-MT |
| PDF Generation | ReportLab |
| Backend | Python, Hugging Face Transformers |

---

## 🚀 Quick Start

### 🔹 Prerequisites
- Python **3.8+**
- Git
- FFmpeg (required for audio processing)

---

### 🔹 Installation

Clone the repository:
```bash
git clone https://github.com/your-username/LectNotes-AI.git
cd LectNotes-AI
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
streamlit run app.py
```


### 📋 Usage Workflow
Upload a lecture audio file (MP3 / WAV / M4A)

Convert audio to text using Whisper

View the complete transcript

Generate an AI-powered summary

Translate to Hindi (optional)

Download transcript or summary as PDF

### 🌐 Deployment
## ✅ Streamlit Community Cloud (Recommended)
Push your code to GitHub

Visit https://streamlit.io/cloud

Connect your GitHub repository

Select app.py as the main file

Deploy 🎉

## Other Deployment Options
Render

Railway

Docker

### 📁 Project Structure
``` bash
LectNotes-AI/
│
├── app.py                 # Main Streamlit application
├── utils/
│   ├── audio_processor.py # Audio handling & Whisper integration
│   ├── summarizer.py      # Text summarization logic
│   ├── translator.py      # Translation functions
│   └── pdf_generator.py   # PDF export functionality
├── requirements.txt       # Python dependencies
├── .streamlit/            # Streamlit configuration
├── README.md
├── LICENSE
└── .gitignore
```

🎓 Academic Use Cases

AI/ML Coursework – Convert lectures into revision notes

Multilingual Learning – Hindi translations for regional students

Research – Quick transcription of interviews and seminars

Study Groups – Share summarized lecture content

Accessibility – Audio-to-text support for hearing-impaired users

⚠️ Known Limitations

Translation accuracy may vary for technical terminology

Large audio files (>30 minutes) may take 2–5 minutes to process

Internet connection required for initial model downloads

Best results achieved with clear audio quality

### 🔮 Future Enhancements

🧠 Quiz and question generation from lecture content

🌍 Support for additional Indian languages (Tamil, Telugu, etc.)

✨ Keyword extraction and highlights

📊 Speaker diarization for multi-speaker lectures

🎯 Advanced PDF formatting with table of contents

## 📊 Performance Benchmarks
| Feature        | Processing Time (5-min audio) | Accuracy                |
| -------------- | ----------------------------- | ----------------------- |
| Speech-to-Text | ~45 seconds                   | 95%+                    |
| Summarization  | ~8 seconds                    | Contextually accurate   |
| Translation    | ~5 seconds                    | High for standard terms |


## 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create a feature branch
```bash
git checkout -b feature/AmazingFeature
```
3. Commit your changes
```bash
git commit -m "Add AmazingFeature"
```
4. Push to the branch

```bash
git push origin feature/AmazingFeature
```
5. Open a Pull Request

📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.

👩‍💻 Author

Sanjana Krishnan
LinkedIn | GitHub

Building AI tools for education 🚀

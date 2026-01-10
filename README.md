🎧 LectNotes AI
Smart Lecture Assistant for Audio-to-Notes & Summaries

LectNotes AI is an AI-powered web application that converts lecture audio into structured transcripts, concise summaries, and multilingual outputs. It is designed for students, educators, and multilingual learners who want to automate lecture note creation efficiently.

✨ Key Features
🎤 Speech-to-Text Conversion

Accurately converts lecture audio files (MP3, WAV, M4A) into text using OpenAI Whisper.

📘 AI-Powered Summarization

Generates concise and meaningful summaries from long transcripts using DistilBART.

🌐 Multilingual Support (English & Hindi)

English audio → English text

Hindi audio → Hindi text

Optional English ↔ Hindi translation

📄 PDF Export

Download transcripts and summaries as professionally formatted PDFs for offline study.

🎨 Interactive User Interface

Clean, academic-friendly interface built using Streamlit for a smooth user experience.

🛠️ Tech Stack
Category	Technology
Framework	Streamlit
Speech-to-Text	OpenAI Whisper
Summarization	DistilBART (Hugging Face)
Translation	Helsinki NLP Opus-MT
PDF Generation	ReportLab
Backend	Python, Hugging Face Transformers
🚀 Quick Start
🔹 Prerequisites

Python 3.8+

Git

FFmpeg (required for audio processing)

🔹 Installation

Clone the repository:

git clone https://github.com/your-username/LectNotes-AI.git
cd LectNotes-AI


Install dependencies:

pip install -r requirements.txt


Run the application:

streamlit run app.py

📋 Usage Workflow

Upload a lecture audio file (MP3 / WAV / M4A)

Convert audio to text using Whisper

View the complete transcript

Generate an AI-powered summary

Translate to Hindi (optional)

Download transcript or summary as PDF

🌐 Deployment
✅ Streamlit Community Cloud (Recommended)

Push your code to GitHub

Visit 👉 https://streamlit.io/cloud

Connect your GitHub repository

Select app.py as the main file

Deploy 🎉

Other Supported Platforms

Render

Railway

Docker (Dockerfile compatible)

📁 Project Structure
LectNotes-AI/
│
├── app.py                 # Main Streamlit application
├── utils/
│   ├── audio_processor.py # Whisper & audio handling
│   ├── summarizer.py      # Text summarization logic
│   ├── translator.py      # Translation functions
│   └── pdf_generator.py   # PDF export utilities
├── requirements.txt       # Python dependencies
├── .streamlit/            # Streamlit configuration
├── README.md
├── LICENSE
└── .gitignore

🎓 Academic Use Cases

AI/ML Coursework – Convert lectures into revision notes

Multilingual Learning – Hindi support for regional learners

Research – Transcribe interviews, seminars, and talks

Study Groups – Share summarized lecture content

Accessibility – Audio-to-text for hearing-impaired users

⚠️ Known Limitations

Translation accuracy may vary for technical terminology

Large audio files (>30 minutes) may take 2–5 minutes to process

Initial model download requires internet access

Best results achieved with clear audio quality

🔮 Future Enhancements

🧠 Quiz & question generation from lecture content

🌍 Support for additional Indian languages (Tamil, Telugu, etc.)

✨ Keyword extraction & highlights

📊 Speaker diarization (multi-speaker lectures)

🎯 Advanced PDF formatting (TOC, sections)

📊 Performance Benchmarks
Feature	Processing Time (5-min audio)	Accuracy
Speech-to-Text	~45 seconds	95%+
Summarization	~8 seconds	Contextually accurate
Translation	~5 seconds	High (standard terms)
🤝 Contributing

Contributions are welcome!

Fork the repository

Create a feature branch

git checkout -b feature/AmazingFeature


Commit your changes

git commit -m "Add AmazingFeature"


Push to the branch

git push origin feature/AmazingFeature


Open a Pull Request

📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.

👩‍💻 Author

Sanjana Krishnan
📎 LinkedIn | GitHub
Building AI tools for education 🚀

⭐ Support the Project

If this project helped you, please consider giving it a ⭐ on GitHub!

📌 Request a feature | 🐞 Report a bug

<div align="center"> <img src="https://img.shields.io/badge/Built%20With-Streamlit-orange.svg" /> <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg" /> </div>

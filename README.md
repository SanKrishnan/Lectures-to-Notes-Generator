# 🎧 **LectNotes AI**
### *Smart Lecture Assistant for Audio-to-Notes & Summaries*

LectNotes AI is an AI-powered web application that converts lecture audio into structured text and concise summaries. It also supports English-to-Hindi translation, making it useful for multilingual learners and academic environments.

---

## 🚀 **Key Features**

- 🎤 **Speech-to-Text Conversion**  
  Converts uploaded lecture audio files into readable text using the Whisper speech recognition model.

- 📘 **Automatic Lecture Summarization**  
  Generates concise and meaningful summaries from long lecture transcripts using transformer-based NLP models.

- 🌐 **English → Hindi Translation**  
  Provides multilingual support by translating transcripts and summaries into Hindi.

- 📄 **PDF Export**  
  Enables users to download transcripts and summaries as PDF files for offline study.

- 🎨 **Clean & Interactive UI**  
  Built using Streamlit with a simple, academic-friendly interface.

---

## 🧠 **Technologies Used**

- **Python**
- **Streamlit** – Web application framework  
- **OpenAI Whisper** – Speech-to-text model  
- **DistilBART** – Text summarization model  
- **Helsinki NLP Opus-MT** – English to Hindi translation  
- **Hugging Face Transformers**
- **ReportLab** – PDF generation  

---

## 🏗️ **System Workflow**

1. User uploads a lecture audio file  
2. Audio is converted to text using Whisper  
3. Transcript is summarized using a transformer model  
4. Optional translation to Hindi is applied  
5. Output is displayed and can be downloaded as a PDF  

---

## 📁 **Project Structure**
LectNotes-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

---
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/LectNotes-AI.git
cd LectNotes-AI
```

2️⃣ Install Dependencies
```bash
Copy code
pip install -r requirements.txt
```

3️⃣ Run the Application
```bash
Copy code
streamlit run app.py
```

🌍 Deployment

LectNotes AI can be deployed using Streamlit Community Cloud:

Push the project to GitHub

Visit https://streamlit.io/cloud

Connect your GitHub repository

Select app.py as the main file

Click Deploy


🎓 Academic Use Case

This project is suitable for:

AI & Data Science coursework

Lecture note automation

Multilingual education support

Speech and NLP-based academic applications


⚠️ Limitations

Translation accuracy may vary for technical terms

Large audio files may take longer to process

Internet connection required for model loading


🔮 Future Enhancements

Quiz generation from lecture content

Support for additional languages

Keyword extraction and highlights

Improved formatting for generated PDFs


👩‍💻 Author

Sanjana Krishnan


📜 License

This project is licensed under the MIT License and is intended for academic and educational use.

🎧 LectNotes AI – Smart Lecture Assistant

LectNotes AI is an AI-powered web application that converts lecture audio into readable text and concise summaries. It also supports English-to-Hindi translation to help multilingual learners understand academic content more effectively.

✨ Key Features

🎤 Speech-to-Text Conversion
Converts lecture audio files into text using the Whisper speech recognition model.

📘 Automatic Lecture Summarization
Generates concise summaries from long lecture transcripts using transformer-based NLP models.

🌐 English → Hindi Translation
Supports multilingual learning by translating transcripts and summaries into Hindi.

📄 PDF Download
Allows users to download transcripts and summaries as PDF files.

🎨 Simple & Interactive UI
Built using Streamlit with a clean and user-friendly interface.

🧠 Technologies Used

Python

Streamlit – Web interface

OpenAI Whisper – Speech-to-text

DistilBART – Text summarization

Helsinki NLP Opus-MT – English to Hindi translation

ReportLab – PDF generation

Hugging Face Transformers

🏗️ System Workflow

User uploads a lecture audio file

Audio is converted into text using Whisper

Text is summarized using a transformer model

Output can be translated to Hindi

Results are displayed and downloadable as PDFs

📁 Project Structure
LectNotes-AI/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE

⚙️ How to Run Locally

1️⃣ Clone the repository:

git clone https://github.com/your-username/LectNotes-AI.git
cd LectNotes-AI


2️⃣ Install dependencies:

pip install -r requirements.txt


3️⃣ Run the application:

streamlit run app.py

🌍 Deployment

The project can be deployed on Streamlit Community Cloud by connecting the GitHub repository and selecting app.py as the main file.

🎓 Academic Use

This project is suitable for:

AI & Data Science coursework

Lecture note automation

Multilingual education support

Speech and NLP-based applications

⚠️ Limitations

Translation accuracy may vary for technical terms

Large audio files may take longer to process

Requires internet access for model loading

🔮 Future Scope

Quiz generation from lecture content

Support for additional languages

Keyword extraction and highlights

Improved formatting of generated PDFs

👩‍💻 Author

Sanjana Krishnan



📜 License

This project is licensed under the MIT License and is intended for academic and educational use.

🧠 Resume Screening (Flask App)

A full-featured Flask web application that automatically extracts text from uploaded PDF/DOCX resumes, analyzes their relevance to a given Job Description (JD) using TF-IDF + Logistic Regression, and performs sentiment analysis on introductions or cover texts.
It exports results to CSV/Excel, allowing recruiters to quickly rank candidates based on skill match, relevance, and tone.

🎯 Built to streamline candidate shortlisting and provide data-driven insights during hiring.

✨ Features

📄 Upload multiple resumes (PDF or DOCX)

🧩 Automatic text extraction from resumes

⚙️ Relevance prediction using Logistic Regression (TF-IDF)

💬 Sentiment analysis using NLTK’s VADER

📊 Similarity score between resume & JD

📁 CSV/Excel export for saved candidate results

🌐 REST API endpoints for integration

🧱 Simple Flask UI (Jinja templates)

🧰 Tech Stack

Backend	-: Python, Flask

Machine Learning / NLP	scikit-learn, NLTK, TextBlob

Data Handling -: pandas, numpy

File Parsing -:	pdfplumber, python-docx, PyPDF2

Storage -:	CSV (default)

Frontend -:	HTML, CSS, JS 

🚀 Getting Started

1️⃣ Clone the repository
Download zip and extract the project files 
cd resume-screening

2️⃣ Create and activate a virtual environment
python -m venv .venv

3️⃣ Install dependencies
Create requirements.txt or install directly:
pip install flask scikit-learn numpy pandas nltk textblob pdfplumber PyPDF2 python-docx openpyxl
(If using NLTK for the first time, download the required data:)
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

4️⃣ Run the Flask app
python app.p
Then open the app in your browser:
http://127.0.0.1:5000/

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pdfplumber
import docx
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------- NLTK Setup --------------------
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# -------------------- Flask App ---------------------
app = Flask(__name__)
os.makedirs("uploads", exist_ok=True)

csv_file = "candidates_extended_dataset1.csv"

# -------------------- Load Dataset ------------------
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, dtype=str)
else:
    df = pd.DataFrame(columns=[
        "CID", "NAME", "SKILLS", "EXPERIENCE", 
        "CERTIFICATION", "JOB_DESCRIPTION", 
        "RELEVANT", "SENTIMENT_SCORE"
    ])

# Clean data safely
if "RELEVANT" in df.columns:
    df["RELEVANT"] = pd.to_numeric(df["RELEVANT"], errors="coerce").fillna(0).astype(int)
else:
    df["RELEVANT"] = 0

df = df.fillna("")

# -------------------- Model Setup -------------------
vectorizer = TfidfVectorizer()
model = LogisticRegression()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

# Train model only if we have valid labeled data
if not df.empty and "SKILLS" in df.columns and "CERTIFICATION" in df.columns:
    X_data = (df["SKILLS"] + " " + df["CERTIFICATION"] + " " + df["JOB_DESCRIPTION"]).fillna("")
    y_data = df["RELEVANT"]

    if len(df) > 2 and len(y_data.unique()) > 1:
        X = vectorizer.fit_transform(X_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("\n✅ Model Trained Successfully!")
        print(f"Accuracy:  {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")
    else:
        print("\n⚠️ Not enough data or only one class in RELEVANT column. Model not trained.")
else:
    print("\n⚠️ Empty dataset or missing required columns. Model not trained.")

# -------------------- Helper Functions --------------------
def extract_text_from_resume(file_path):
    """Extract text from PDF or DOCX resume."""
    text = ""
    try:
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            text = " ".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""
    return text.strip()

def analyze_sentiment(resume_text, job_desc, skills):
    """Analyze sentiment around skills within resume."""
    if not resume_text or skills == "None":
        return 0.0

    job_tokens = [w.lower() for w in word_tokenize(job_desc) if w.lower() not in stop_words]
    resume_tokens = word_tokenize(resume_text.lower())
    skill_words = skills.lower().split(", ")
    context_sentences = []

    for skill in skill_words:
        for i, token in enumerate(resume_tokens):
            if skill in token:
                start, end = max(0, i - 10), min(len(resume_tokens), i + 10)
                context_sentences.append(" ".join(resume_tokens[start:end]))

    if not context_sentences:
        return 0.0

    total_sentiment = sum(sia.polarity_scores(s)["compound"] for s in context_sentences)
    avg_sentiment = total_sentiment / len(context_sentences)
    return round(((avg_sentiment + 1) / 2) * 100, 2)

def extract_features_with_sentiment(text, job_desc):
    """Extract candidate name, skills, experience, certifications, and sentiment."""
    name_pattern = r"(?i)(?:name[:\s]*)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    skills_pattern = r"(Python|Java|C\+\+|JavaScript|React|Node\.js|SQL|Machine Learning|AWS)"
    experience_pattern = r"(\d+)\s*years?"
    certification_pattern = r"(Certified in \w+)"

    name_match = re.findall(name_pattern, text)
    name = name_match[0] if name_match else "Unknown"
    skills = ", ".join(re.findall(skills_pattern, text)) or "None"
    experience = int(re.findall(experience_pattern, text)[0]) if re.findall(experience_pattern, text) else 0
    certifications = ", ".join(re.findall(certification_pattern, text)) or "None"
    sentiment_score = analyze_sentiment(text, job_desc, skills)
    return name, skills, experience, certifications, sentiment_score

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/speech")
def speech():
    return render_template("speechresume.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    global df

    if "resumes" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    job_desc = request.form.get("job_desc", "").strip()
    if not job_desc:
        return jsonify({"error": "Job description is required"}), 400

    files = request.files.getlist("resumes")
    results = []

    for file in files:
        if not file or file.filename == "":
            continue

        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        resume_text = extract_text_from_resume(file_path)
        if not resume_text:
            results.append({
                "filename": file.filename,
                "error": "Could not extract text",
                "score": 0,
                "sentiment_score": 0
            })
            continue

        name, skills, experience, certification, sentiment_score = extract_features_with_sentiment(resume_text, job_desc)

        # Predict relevance (if model trained)
        if not df.empty and len(df) > 2 and "SKILLS" in df.columns:
            try:
                resume_vector = vectorizer.transform([skills + " " + certification])
                relevance = int(model.predict(resume_vector)[0])
            except Exception:
                relevance = 0
        else:
            relevance = 0

        # ✅ Always compute similarity score
        try:
            tfidf = TfidfVectorizer().fit_transform([resume_text, job_desc])
            similarity_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            similarity_score = round(similarity_score * 100, 2)
        except Exception as e:
            print(f"Error computing similarity: {e}")
            similarity_score = 0

        # Unique Candidate ID generation
        try:
            last_id = df["CID"].iloc[-1]
            prefix = ''.join(filter(str.isalpha, last_id))
            number = ''.join(filter(str.isdigit, last_id))
            unique_id = f"{prefix}{int(number) + 1}" if number else "C100"
        except Exception:
            unique_id = "C100"

        new_entry = pd.DataFrame([{
            "CID": unique_id,
            "NAME": name,
            "SKILLS": skills,
            "EXPERIENCE": experience,
            "CERTIFICATION": certification,
            "JOB_DESCRIPTION": job_desc,
            "RELEVANT": relevance,
            "SENTIMENT_SCORE": sentiment_score
        }])

        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(csv_file, index=False)

        results.append({
            "filename": file.filename,
            "CID": unique_id,
            "name": name,
            "skills": skills,
            "experience": experience,
            "certification": certification,
            "relevance": relevance,
            "score": similarity_score,
            "sentiment_score": sentiment_score
        })

    results = sorted(results, key=lambda x: x["score"] + x["sentiment_score"], reverse=True)
    return jsonify({"resumes": results})

# -------------------- Speech Intro Route --------------------
@app.route("/speech_intro_analyze", methods=["POST"])
def speech_intro_analyze():
    data = request.get_json()
    job_desc = data.get("job_desc", "").strip()
    intro_text = data.get("intro_text", "").strip()

    if not job_desc or not intro_text:
        return jsonify({"error": "Job description and introduction text are required"}), 400

    name, skills, experience, certification, sentiment_score = extract_features_with_sentiment(intro_text, job_desc)

    try:
        tfidf = TfidfVectorizer().fit_transform([intro_text, job_desc])
        similarity_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        similarity_score = round(similarity_score * 100, 2)
    except Exception as e:
        print(f"Error computing similarity: {e}")
        similarity_score = 0

    return jsonify({
        "intro_score": similarity_score,
        "sentiment_score": sentiment_score
    })

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)

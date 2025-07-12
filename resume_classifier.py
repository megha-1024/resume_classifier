import re
import nltk
import pandas as pd
import streamlit as st
import pdfplumber
import docx2txt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

nltk.download("stopwords")
nltk.download("wordnet")

#NLP Preprocessing 
stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()

def clean_resume(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

#Skill Extraction 
skills_set = [
    "python", "java", "c++", "sql", "excel", "tableau", "power bi", "pandas", "numpy",
    "machine learning", "deep learning", "nlp", "django", "flask", "react", "html",
    "css", "javascript", "data analysis", "tensorflow", "keras", "git", "github",
    "docker", "kubernetes", "aws", "azure", "linux", "spark", "hadoop", "problem solving",
    "communication", "leadership", "time management", "project management", "data visualization"
]

def extract_skills(text):
    text = text.lower()
    extracted = [skill for skill in skills_set if skill in text]
    return list(set(extracted))

#Recommended Skills Map 
job_skills_map = {
    "Data Scientist": [
        "python", "pandas", "numpy", "machine learning", "data analysis", "tensorflow", "keras", "sql", "matplotlib", "seaborn"
    ],
    "Web Developer": [
        "html", "css", "javascript", "react", "flask", "django", "git", "github"
    ],
    "HR": [
        "communication", "leadership", "time management", "excel", "project management"
    ],
    "Software Engineer": [
        "python", "java", "c++", "git", "problem solving", "sql", "linux", "github"
    ],
    "UI/UX Designer": [
        "figma", "adobe xd", "html", "css", "communication", "user research"
    ],
    "Project Manager": [
        "project management", "time management", "leadership", "excel", "communication", "jira"
    ],
    "Business Analyst": [
        "excel", "sql", "power bi", "data analysis", "communication", "tableau"
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "linux", "git", "github", "jenkins"
    ],
    "Cybersecurity Analyst": [
        "linux", "networking", "siem", "python", "communication", "problem solving"
    ]
}

def recommend_missing_skills(predicted_role, extracted_skills):
    expected = job_skills_map.get(predicted_role, [])
    missing = [skill for skill in expected if skill not in extracted_skills]
    return missing

#Load Dataset & Train Model 
df = pd.read_csv('C:\\Users\\Megha\\OneDrive\\Documents\\achive and projects\\AI based resume classifier\\resume_dataset.csv')
df["cleaned"] = df["Resume"].apply(clean_resume)

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["cleaned"]).toarray()
y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#Streamlit Web App 
def predict_resume_category(resume_text):
    cleaned = clean_resume(resume_text)
    vec = tfidf.transform([cleaned])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    return prediction, probabilities

# Streamlit UI
st.set_page_config(page_title="AI Resume Classifier", layout="centered")
st.title("🧠 AI-Powered Resume Classifier")

uploaded_file = st.file_uploader("📤 Upload your resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "".join(page.extract_text() or '' for page in pdf.pages)
    elif file_type == "docx":
        text = docx2txt.process(uploaded_file)
    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")

    st.text_area("📄 Extracted Resume Text:", value=text, height=300)

    if st.button("🔍 Predict Job Role & Analyze Skills"):
        predicted_role, probabilities = predict_resume_category(text)
        st.success(f"🎯 Predicted Job Category: **{predicted_role}**")

        # Confidence chart
        labels = model.classes_
        fig, ax = plt.subplots()
        sns.barplot(x=probabilities, y=labels, palette="viridis", ax=ax)
        ax.set_title("Prediction Confidence (%)")
        ax.set_xlabel("Confidence")
        st.pyplot(fig)

        # Skills found
        skills_found = extract_skills(text)
        st.subheader("🛠️ Skills Detected:")
        if skills_found:
            st.write(", ".join(skills_found))
        else:
            st.write("No recognized skills found.")

        # Missing skills
        recommended = recommend_missing_skills(predicted_role, skills_found)
        st.subheader("📌 Recommended Skills to Add:")
        if recommended:
            st.warning(", ".join(recommended))
        else:
            st.success("Your resume already covers the key skills for this role!")

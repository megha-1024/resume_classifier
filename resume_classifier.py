import re
import nltk
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

nltk.download("stopwords")
nltk.download("wordnet")

#Preprocessing Function
stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()

def clean_resume(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

#Loading Dataset 
df = pd.read_csv("resume_dataset.csv")  
df.rename(columns={df.columns[0]: "Resume", df.columns[1]: "Category"}, inplace=True)

#Clean Resumes 
df["cleaned"] = df["Resume"].apply(clean_resume)

#Feature Extraction 
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["cleaned"]).toarray()
y = df["Category"]

#Train-Test Split and Model Training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Step 9: Streamlit Interface ---
import pdfplumber
import docx2txt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def predict_resume_category(resume_text):
    cleaned = clean_resume(resume_text)
    vec = tfidf.transform([cleaned])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    return prediction, probabilities


#App Layout 
st.title("AI-Powered Resume Classifier")
resume_input = st.text_area("Paste your resume here")
if st.button("Predict Job Role"):
    predicted_role = predict_resume_category(resume_input)
    st.success(f"Predicted Job Category: {predicted_role}")

st.text("OR")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx", "txt"])
text = ""

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join(page.extract_text() or '' for page in pdf.pages)
    elif file_type == "docx":
        text = docx2txt.process(uploaded_file)
    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")

    st.text_area("Extracted Resume Text", value=text, height=300)

    if st.button("Job role"):
        predicted_role, probabilities = predict_resume_category(text)
        st.success(f"Predicted Job Category: {predicted_role}")

        #confidence chart
        labels = model.classes_
        fig, ax = plt.subplots()
        sns.barplot(x=probabilities, y=labels, palette="viridis", ax=ax)
        ax.set_title("Prediction Confidence (%)")
        ax.set_xlabel("Confidence")
        st.pyplot(fig)



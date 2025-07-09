**AI-Powered Resume Classifier**

A smart web app built with Streamlit that classifies resumes into job categories like **Data Scientist**, **Web Developer**, **HR**, and more.
Users can either **paste their resume text** or **upload PDF/DOCX files** to receive predictions instantly.

Built for aspiring data scientists and job-seekers to get quick insights into how their resume is perceived by AI.


**Features**
-  Upload resumes in PDF, DOCX, or TXT format
-  Paste plain resume text directly
-  Predicts job role using trained ML model
-  Displays confidence scores for each prediction
-  Hosted using [Streamlit Cloud](https://streamlit.io/cloud)


**Tech Stack**
 Tool/Library  -    Purpose                         
 Python       -  Core programming language        
 Streamlit    -  Interactive web interface        
 scikit-learn -  Model training & classification  
 pandas       -  Data handling                    
 nltk         -  Text preprocessing               
 TfidfVectorizer - Feature extraction from text     
 matplotlib/seabor - Confidence visualization      
 pdfplumber / docx2txt - File text extraction        

**Project Structure**
├── resume_classifier.py (Main Streamlit app)

├── resume_dataset_extended.csv (Training dataset)

├── requirements.txt (Project dependencies)

└── README.md (Project documentation)

**Dataset:**
This app uses a custom dataset with labeled resumes for 9 roles:
-Data Scientist
-Web Developer
-HR
-Software Engineer
-UI/UX Designer
-Project Manager
-Business Analyst
-DevOps Engineer
-Cybersecurity Analyst


**Model Used**
-TF-IDF Vectorizer for resume text embedding
-Logistic Regression for classification

**Live Demo**
Try it here:
https://resume-ai-classifier.streamlit.app/



import os
import sys
import pandas as pd
import joblib
import string
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')


stop_words = set(stopwords.words('english'))
from PyPDF2 import PdfReader


def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        filtered_words = [word for word in tokens if word not in stop_words]
        return ' '.join(filtered_words)
    else:
        return ""

def load_model(model_path='model.pkl'):
    return joblib.load(model_path)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def categorize_resumes(input_dir, model):
    vectorizer = joblib.load('vectorizer.pkl')
    results = []

    categorized_dir = 'categorized_resumes'
    if not os.path.exists(categorized_dir):
        os.makedirs(categorized_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            resume_path = os.path.join(input_dir, filename)
            resume_text = extract_text_from_pdf(resume_path)
            resume_cleaned = preprocess_text(resume_text)
            resume_vectorized = vectorizer.transform([resume_cleaned])
            predicted_category = model.predict(resume_vectorized)[0]

            category_dir = os.path.join(categorized_dir, predicted_category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)

            new_resume_path = os.path.join(category_dir, filename)
            os.rename(resume_path, new_resume_path)
            results.append([filename, predicted_category])

    df_results = pd.DataFrame(results, columns=['Filename', 'Category'])
    df_results.to_csv('categorized_resumes.csv', index=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Error: The directory {input_dir} does not exist.")
        sys.exit(1)

    model = load_model()
    categorize_resumes(input_dir, model)

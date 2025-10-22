# Spam Mail Detector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import string
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
data = pd.read_csv(url, sep='\t', names=['label', 'message'])

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['clean_message'] = data['message'].apply(preprocess_text)

data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    data['clean_message'], data['label_num'],
    test_size=0.2, random_state=42, stratify=data['label_num']
)

vectorizer = TfidfVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("📊 Model Performance:\n")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

def predict_message(msg):
    msg_clean = preprocess_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    prediction = model.predict(msg_vec)[0]
    return "Spam" if prediction == 1 else "Ham"

print("\n🧾 Example Predictions:")
print(predict_message("Congratulations! You've won a free vacation to Bahamas. Call now!"))
print(predict_message("Hey, are we still meeting for lunch today?"))

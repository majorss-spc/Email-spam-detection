import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load dataset (Assuming 'spam.csv' with 'v1' (label) and 'v2' (message))
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels: 'ham' → 0, 'spam' → 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Apply text cleaning to messages
df['message'] = df['message'].apply(clean_text)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['message']).toarray()
y = df['label']

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Function to Predict New Emails
def predict_email(email):
    email = clean_text(email)  # Clean input email
    email_vector = vectorizer.transform([email]).toarray()  # Convert to TF-IDF
    prediction = model.predict(email_vector)[0]  # Predict spam or not
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit App
st.title("Email Spam Detection")

st.write("## Model Evaluation")
st.write(f"**Model Accuracy:** {accuracy}")
st.write("**Classification Report:**")
st.text(classification_rep)

st.write("## Predict New Emails")
email_text = st.text_area("Enter an email to check:")
if st.button("Predict"):
    if email_text:
        prediction = predict_email(email_text)
        st.write(f"**Prediction:** {prediction}")
    else:
        st.write("Please enter an email text to predict.")
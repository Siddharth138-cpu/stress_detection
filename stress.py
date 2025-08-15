import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Download stopwords (only once)
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
stopword = set(stopwords.words('english'))

# Cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stopword]
    return " ".join(words)

# Load and preprocess data
data = pd.read_csv("stress.csv")
data["text"] = data["text"].apply(clean)
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]

# Train model
x = np.array(data["text"])
y = np.array(data["label"])
cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=42)
model = BernoulliNB()
model.fit(xtrain, ytrain)

# Streamlit UI
st.title("🧠 Stress Detection App")
st.write("Enter a text below to check if it indicates stress or not.")

user_input = st.text_area("Your Text", "")

if st.button("Predict"):
    if user_input.strip() != "":
        cleaned = clean(user_input)
        vectorized = cv.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)[0]
        st.subheader(f"Prediction: **{prediction}**")
    else:
        st.warning("Please enter some text to analyze.")


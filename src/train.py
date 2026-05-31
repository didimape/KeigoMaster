# ---------------------------------------------------------
# This file teaches the model using labeled examples
# ---------------------------------------------------------
# Convert text into numbers
# The model learns patterns


import pandas as pd
import joblib

# -----------------------
# TEXT VECTORIZATION
# -----------------------
# Convert Japanese text into numerical vectors
# using TF-IDF feature extraction.

from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------
# MODEL TRAINING
# -----------------------
# Train a Naive Bayes classifier
# to detect Japanese speech styles.

from sklearn.naive_bayes import MultinomialNB

from src.tokenizer import tokenize_japanese

# ------------------------------------
# LOAD DATASET
# ------------------------------------

df = pd.read_csv("data/japanese_sentences.csv")

X = df["例文"]
y = df["分類"]

"""# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)"""
# -----------------------
# TEXT VECTORIZATION
# -----------------------
# Convert Japanese text into numerical vectors.

vectorizer = TfidfVectorizer(
    tokenizer=tokenize_japanese,
    token_pattern=None
)

X_vectorized = vectorizer.fit_transform(X)

# -----------------------
# MODEL TRAINING
# -----------------------
# Train Naive Bayes classifier.

model = MultinomialNB()

model.fit(X_vectorized, y)

# -----------------------
# SAVE MODEL
# -----------------------
# Save trained model and vectorizer.

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model trained successfully!")
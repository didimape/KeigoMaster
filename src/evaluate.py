import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.tokenizer import tokenize_japanese



# -----------------------
# LOAD DATASET
# -----------------------

df = pd.read_csv("data/japanese_sentences.csv")

X = df["例文"]
y = df["分類"]

# -----------------------
# TRAIN / TEST SPLIT
# -----------------------
# Split dataset into training and testing data.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------
# TEXT VECTORIZATION
# -----------------------

vectorizer = TfidfVectorizer(
    tokenizer=tokenize_japanese,
    token_pattern=None
)

X_train_vectorized = vectorizer.fit_transform(X_train)

# -----------------------
# MODEL TRAINING
# -----------------------

model = MultinomialNB()

model.fit(X_train_vectorized, y_train)

# -----------------------
# MODEL EVALUATION
# -----------------------

X_test_vectorized = vectorizer.transform(X_test)

predictions = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)


from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

print(model.predict(vectorizer.transform(["ありがとう"])))
print(model.predict(vectorizer.transform(["お疲れ様でございます"])))
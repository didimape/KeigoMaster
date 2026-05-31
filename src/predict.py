import joblib
import os
# -----------------------
# LOAD ALREADY TRAINED MODEL
# -----------------------

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_sentence(sentence):
    """
    Predict Japanese speech style.
    """

    vector = vectorizer.transform([sentence])

    prediction = model.predict(vector)[0]

    return prediction


# TEST

"""text = "こんにちは、元気ですか"

X = vectorizer.transform([text])
pred = model.predict(X)

print(pred)"""
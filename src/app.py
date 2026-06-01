"""La interfaz:

Streamlit
Flask
etc."""

import tkinter as tk
import joblib
import re

# -----------------------
# LOAD MODEL + VECTORIZER
# -----------------------

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/model.pkl")

# -----------------------
# DETECT JAPANESE
# -----------------------

def is_japanese(text):
    return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text))

# -----------------------
# PREDICTION FUNCTION
# -----------------------

def predict_text():
    text = text_input.get().strip()

    # 1. CHECK EMPTY
    if not text:
        result_label.config(text="❌ Escribe algo")
        return

    # 2. CHECK JAPANESE
    if not is_japanese(text):
        result_label.config(text="❌ No es japonés")
        return

    # 3. VECTORIZE
    X = vectorizer.transform([text])

    # 4. PROBABILITY CHECK (IMPORTANT PART)
    proba = model.predict_proba(X)[0]
    max_prob = max(proba)

    # 5. THRESHOLD FILTER
    if max_prob < 0.60:
        result_label.config(text="❌ No es keigo claro")
        return

    # 6. FINAL PREDICTION
    prediction = model.predict(X)[0]

    result_label.config(text=f"Resultado: {prediction} ({max_prob:.2f})")

# -----------------------
# WINDOW
# -----------------------

window = tk.Tk()

window.title("KeigoMaster")

window.geometry("500x300")

# -----------------------
# TITLE
# -----------------------

title_label = tk.Label(
    window,
    text="KeigoMaster",
    font=("Arial", 20)
)

title_label.pack(pady=10)

# -----------------------
# INPUT
# -----------------------

text_input = tk.Entry(
    window,
    width=50,
    font=("Arial", 14)
)

text_input.pack(pady=10)

# -----------------------
# BUTTON
# -----------------------

predict_button = tk.Button(
    window,
    text="Predecir",
    command=predict_text,
    font=("Arial", 12)
)

predict_button.pack(pady=10)

# -----------------------
# RESULT
# -----------------------

result_label = tk.Label(
    window,
    text="Resultado:",
    font=("Arial", 14)
)

result_label.pack(pady=20)

# -----------------------
# START APP
# -----------------------

window.mainloop()
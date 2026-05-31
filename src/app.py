"""La interfaz:

Streamlit
Flask
etc."""

import tkinter as tk
import joblib

# -----------------------
# LOAD MODEL + VECTORIZER
# -----------------------

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/model.pkl")

# -----------------------
# PREDICTION FUNCTION
# -----------------------

def predict_text():
    text = text_input.get()

    X = vectorizer.transform([text])

    prediction = model.predict(X)

    result_label.config(text=f"Resultado: {prediction[0]}")

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
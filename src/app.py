import tkinter as tk
import joblib
import re

# -----------------------
# LOAD MODEL
# -----------------------

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/model.pkl")

# -----------------------
# JAPANESE CHECK
# -----------------------

def is_japanese(text):
    return bool(re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', text))

def explain_keigo(text, label):
    explanations = []

    # Regla 1: honorífico típico
    if "お" in text or "ご" in text:
        explanations.append("✔ contiene prefijos honoríficos (お / ご)")

    # Regla 2: verbos típicos de respeto
    if "なさる" in text or "いらっしゃる" in text:
        explanations.append("✔ verbo de respeto detectado")

    # Regla 3: 謙譲語 típico
    if "いたします" in text or "申し上げます" in text:
        explanations.append("✔ forma humilde (謙譲語) detectada")

    # Regla 4: formalidad general
    if label == "丁寧語":
        explanations.append("ℹ lenguaje formal básico (丁寧語)")

    if label == "尊敬語":
        explanations.append("ℹ lenguaje de respeto hacia otros (尊敬語)")

    if label == "謙譲語":
        explanations.append("ℹ lenguaje humilde para hablar de uno mismo (謙譲語)")

    return "\n".join(explanations)

# -----------------------
# COLORS
# -----------------------

BG = "#1e1e2e"
CARD = "#2a2a3c"
TEXT = "#ffffff"
ACCENT = "#4cc9f0"
GOOD = "#2ecc71"
BAD = "#e74c3c"
WARN = "#f1c40f"

# -----------------------
# PREDICT
# -----------------------

def predict_text(event=None):
    text = entry.get().strip()

    if not text:
        update_result("Escribe algo", WARN)
        return

    if not is_japanese(text):
        update_result("❌ No es japonés", BAD)
        return

    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    max_prob = max(proba)

    if max_prob < 0.60:
        update_result(f"❌ No es keigo claro ({max_prob:.2f})", WARN)
        return

    pred = model.predict(X)[0]

    explanation = explain_keigo(text, pred)

    final_text = f"{pred} ({max_prob:.2f})\n\n{explanation}"

    update_result(final_text, GOOD)

# -----------------------
# UI UPDATE
# -----------------------

def update_result(text, color):
    result_label.config(text=text, fg=color)
    confidence_bar.config(width=int(300 * get_confidence()))

def get_confidence():
    try:
        text = entry.get().strip()
        if not text:
            return 0
        X = vectorizer.transform([text])
        return max(model.predict_proba(X)[0])
    except:
        return 0

# -----------------------
# WINDOW
# -----------------------

window = tk.Tk()
window.title("KeigoMaster AI")
window.geometry("600x400")
window.configure(bg=BG)

# -----------------------
# TITLE
# -----------------------

title = tk.Label(
    window,
    text="KeigoMaster AI",
    font=("Arial", 22, "bold"),
    bg=BG,
    fg=ACCENT
)
title.pack(pady=20)

subtitle = tk.Label(
    window,
    text="Detector de Keigo Japonés",
    font=("Arial", 12),
    bg=BG,
    fg="#aaaaaa"
)
subtitle.pack()

# -----------------------
# INPUT CARD
# -----------------------

card = tk.Frame(window, bg=CARD, padx=20, pady=20)
card.pack(pady=30)

entry = tk.Entry(
    card,
    width=45,
    font=("Arial", 14)
)
entry.pack(pady=10)

btn = tk.Button(
    card,
    text="Analizar",
    command=predict_text,
    bg=ACCENT,
    fg="black",
    font=("Arial", 12, "bold"),
    relief="flat",
    padx=10,
    pady=5
)
btn.pack(pady=10)



# -----------------------
# RESULT
# -----------------------

result_label = tk.Label(
    window,
    text="Esperando entrada...",
    font=("Arial", 14),
    bg=BG,
    fg=TEXT,
    justify="left"
)
result_label.pack(pady=10)

# -----------------------
# CONFIDENCE BAR (visual)
# -----------------------

bar_bg = tk.Frame(window, bg="#444", width=300, height=10)
bar_bg.pack(pady=10)

confidence_bar = tk.Frame(bar_bg, bg=ACCENT, width=0, height=10)
confidence_bar.pack(side="left")

# -----------------------
# FOOTER
# -----------------------

footer = tk.Label(
    window,
    text="AI Keigo Classifier • v1.0",
    font=("Arial", 10),
    bg=BG,
    fg="#666"
)
footer.pack(side="bottom", pady=10)

# -----------------------
# ENTER KEY SUPPORT
# -----------------------

entry.bind("<Return>", predict_text)

# -----------------------
# START APP
# -----------------------

window.mainloop()
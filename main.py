from src.predict import predict_sentence

# -----------------------
# USER INPUT
# -----------------------

sentence = input("Enter a Japanese sentence: ")

# -----------------------
# PREDICTION
# -----------------------

result = predict_sentence(sentence)

print("Detected speech style:", result)
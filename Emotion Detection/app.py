import os
import re
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: F401  (needed for unpickling)
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_word2vec_bilstm.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "emotion_tokenizer.pkl")

MAX_LEN = 60  # same value used in training
EMOTION_LABELS = {
    0: "Sad",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
}
app = Flask(__name__)

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

OOV_INDEX = tokenizer.word_index.get("<OOV>", 1)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_emotion(sentence):
    cleaned = clean_text(sentence)
    if not cleaned:
        return None
    sequence = tokenizer.texts_to_sequences([cleaned])
    known_words = sum(1 for i in sequence[0] if i != OOV_INDEX)
    total_words = len(sequence[0])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]
    predictions = [
        {"label": int(i), "emotion": EMOTION_LABELS[i], "confidence": round(float(p) * 100, 2)}
        for i, p in enumerate(probabilities)
    ]
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "predictions": predictions,
        "known_words": known_words,
        "total_words": total_words,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    result = predict_emotion(text)
    if result is None:
        return jsonify({"error": "No English letters found. Please enter a sentence in English."}), 400
    if result["known_words"] == 0:
        return jsonify({"error": "None of the words were recognised. Try a different sentence."}), 422

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
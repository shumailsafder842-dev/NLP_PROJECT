import os
import re
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: F401  (needed for unpickling)
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Language_Identification_LSTM.keras")
OBJECTS_PATH = os.path.join(BASE_DIR, "language_objects.pkl")

LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

app = Flask(__name__)

# ---- Load model + preprocessing objects once at startup -------------------
model = tf.keras.models.load_model(MODEL_PATH)
with open(OBJECTS_PATH, "rb") as f:
    objects = pickle.load(f)

tokenizer = objects["tokenizer"]
label_encoder = objects["label_encoder"]
MAX_LEN = objects["max_len"]


def tokenize_text(text):
    """Must match the tokenizer used during training in the notebook."""
    text = str(text).lower()
    return re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)


def predict_language(sentence, top_k=3):
    tokens = tokenize_text(sentence)
    if not tokens:
        return None

    text = " ".join(tokens)
    sequence = tokenizer.texts_to_sequences([text])

    # If none of the words were seen in training, the sequence is empty.
    if not sequence[0]:
        return {"known_words": 0, "total_words": len(tokens), "predictions": []}

    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")
    probabilities = model.predict(padded, verbose=0)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_k]
    predictions = []
    for idx in top_indices:
        code = label_encoder.inverse_transform([int(idx)])[0]
        predictions.append(
            {
                "code": code,
                "language": LANGUAGE_NAMES.get(code, code),
                "confidence": round(float(probabilities[idx]) * 100, 2),
            }
        )

    return {
        "known_words": len(sequence[0]),
        "total_words": len(tokens),
        "predictions": predictions,
    }


@app.route("/")
def index():
    languages = sorted(LANGUAGE_NAMES.values())
    return render_template("index.html", languages=languages)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    result = predict_language(text)
    if result is None:
        return jsonify({"error": "No letters found in the text. Please enter a sentence."}), 400
    if not result["predictions"]:
        return jsonify({"error": "None of the words were recognised by the model. Try a longer sentence."}), 422

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
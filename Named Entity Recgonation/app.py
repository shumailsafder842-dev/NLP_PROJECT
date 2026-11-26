"""
Named Entity Recognition web app (BiLSTM trained on CoNLL-2003).

Run the notebook first so these files exist next to this script:
    NER_BiLSTM.keras      - trained model
    NER_word2idx.pkl      - vocabulary
    NER_label_names.json  - label list

Then:
    pip install flask tensorflow numpy
    python app.py
and open http://127.0.0.1:5000
"""

import json
import os
import pickle
import re

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "NER_BiLSTM.keras")
VOCAB_PATH = os.path.join(BASE_DIR, "NER_word2idx.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "NER_label_names.json")

MAX_LEN = 120  # must match the notebook
MAX_CHARS = 2000

DEFAULT_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

app = Flask(__name__)

model = None
word2idx = None
label_names = DEFAULT_LABELS
load_error = None


def load_artifacts():
    """Load the model, vocabulary and labels once at start-up."""
    global model, word2idx, label_names, load_error
    try:
        from tensorflow.keras.models import load_model

        model = load_model(MODEL_PATH)
        with open(VOCAB_PATH, "rb") as f:
            word2idx = pickle.load(f)
        with open(LABELS_PATH) as f:
            label_names = json.load(f)
        print("Model loaded.")
    except Exception as exc:  # missing files, TensorFlow not installed, etc.
        model = None
        load_error = f"{type(exc).__name__}: {exc}"
        print(f"Model not loaded ({load_error}). Running in demo mode.")


# CoNLL-style tokenisation: words (with inner hyphens/apostrophes) and single punctuation marks.
TOKEN_RE = re.compile(r"\w+(?:[-'’.]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def predict_with_model(words):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    unk = word2idx["<UNK>"]
    ids = [word2idx.get(w, unk) for w in words]
    padded = pad_sequences([ids], maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    tags, confs = [], []
    for i in range(len(words)):
        if i < MAX_LEN:
            k = int(np.argmax(probs[i]))
            tags.append(label_names[k])
            confs.append(float(probs[i][k]))
        else:  # beyond what the model sees
            tags.append("O")
            confs.append(0.0)
    return tags, confs


# --- Demo fallback so the page is usable before the model files exist -------------------
_DEMO_LOC = {"london", "paris", "pakistan", "islamabad", "lahore", "karachi", "germany", "france",
             "china", "india", "japan", "tokyo", "berlin", "europe", "asia", "america", "canada"}
_DEMO_ORG = {"microsoft", "google", "apple", "amazon", "facebook", "meta", "reuters", "un", "nato",
             "openai", "anthropic", "tesla", "fifa", "who"}
_STOP = {"The", "A", "An", "In", "On", "At", "He", "She", "It", "They", "We", "I", "And", "But", "Or"}


def predict_demo(words):
    tags = ["O"] * len(words)
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in _DEMO_LOC:
            tags[i] = "B-LOC"
        elif lw in _DEMO_ORG:
            tags[i] = "B-ORG"
        elif w[:1].isupper() and w not in _STOP and w.isalpha():
            prev = tags[i - 1] if i else "O"
            tags[i] = "I-PER" if prev in ("B-PER", "I-PER") else "B-PER"
    return tags, [0.0] * len(words)


def group_entities(tokens, tags, confs, text):
    """Merge BIO tags into entity spans with character offsets."""
    entities, cur = [], None
    for (word, start, end), tag, conf in zip(tokens, tags, confs):
        if tag == "O":
            cur = None
            continue
        prefix, kind = tag.split("-", 1)
        if prefix == "B" or cur is None or cur["type"] != kind:
            cur = {"type": kind, "start": start, "end": end, "confs": [conf]}
            entities.append(cur)
        else:
            cur["end"] = end
            cur["confs"].append(conf)
    return [
        {
            "text": text[e["start"]:e["end"]],
            "type": e["type"],
            "start": e["start"],
            "end": e["end"],
            "confidence": round(sum(e["confs"]) / len(e["confs"]), 4),
        }
        for e in entities
    ]


@app.get("/")
@app.get("/index.html")
def index():
    if not os.path.exists(os.path.join(BASE_DIR, "index.html")):
        return (f"index.html was not found. Put it in the same folder as app.py: {BASE_DIR}", 404)
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/status")
def status():
    return jsonify({"model_loaded": model is not None, "labels": label_names, "max_tokens": MAX_LEN})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Enter some text to analyse."}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Text is too long. Keep it under {MAX_CHARS} characters."}), 400

    tokens = tokenize(text)
    words = [t[0] for t in tokens]
    if model is not None:
        tags, confs = predict_with_model(words)
    else:
        tags, confs = predict_demo(words)

    return jsonify({
        "text": text,
        "demo": model is None,
        "truncated": len(words) > MAX_LEN,
        "tokens": [
            {"text": w, "start": s, "end": e, "tag": t, "confidence": round(c, 4)}
            for (w, s, e), t, c in zip(tokens, tags, confs)
        ],
        "entities": group_entities(tokens, tags, confs, text),
    })


load_artifacts()

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
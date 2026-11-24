
import os
import re
import pickle
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration
MAX_LEN = 40
DECODER_LEN = MAX_LEN - 1

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "CustomerSupportChatbot.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "CustomerSupportChatbot.pkl")

app = Flask(__name__)

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)

# Load tokenizer
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Check special tokens
if "startseq" not in tokenizer.word_index:
    raise ValueError("startseq token is missing from tokenizer.")

START_ID = tokenizer.word_index["startseq"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_response(question):

    question = clean_text(question)

    if not question:
        return "Please enter a valid question."

    # Convert question into sequence
    sequence = tokenizer.texts_to_sequences([question])

    # Encoder input
    encoder_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    # Decoder starts with startseq
    decoder_sequence = [START_ID]

    result = []

    for _ in range(DECODER_LEN):

        decoder_pad = pad_sequences(
            [decoder_sequence],
            maxlen=DECODER_LEN,
            padding="post",
            truncating="post"
        )

        # Predict next word
        prediction = model.predict(
            [encoder_sequence, decoder_pad],
            verbose=0
        )

        # Check model output
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]

        if prediction.ndim != 3:
            raise ValueError(
                f"Expected 3D model output, got {prediction.shape}"
            )

        position = len(decoder_sequence) - 1

        if position >= prediction.shape[1]:
            break

        predicted_id = int(
            np.argmax(prediction[0, position, :])
        )

        word = tokenizer.index_word.get(predicted_id, "")

        # Stop generation
        if word in ("endseq", "startseq", "<OOV>") or not word:
            break

        result.append(word)
        decoder_sequence.append(predicted_id)

    if not result:
        return "Sorry, I could not generate a response."

    reply = " ".join(result)

    return reply[0].upper() + reply[1:]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({
            "reply": "Please type a message first."
        }), 400

    try:
        response = generate_response(message)

        return jsonify({
            "reply": response
        })

    except Exception as e:
        app.logger.exception("Chatbot prediction error")

        return jsonify({
            "reply": "Sorry, something went wrong while generating the response."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)

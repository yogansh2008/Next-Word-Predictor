import gradio as gr
import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load trained model
model = load_model("next_word_model_new.keras")

# Reverse dictionary
idx2word = {v: k for k, v in tokenizer.word_index.items()}

MAX_LEN = 5

def predict(text):
    if not text.strip():
        return "Please enter some text."

    seq = tokenizer.texts_to_sequences([text.lower()])
    seq = pad_sequences(seq, maxlen=MAX_LEN, padding="pre")

    probs = model.predict(seq, verbose=0)[0]
    top_ids = np.argsort(probs)[-5:][::-1]

    result = []
    for idx in top_ids:
        word = idx2word.get(idx, "Unknown")
        result.append(f"{word} ({probs[idx]*100:.2f}%)")

    return "\n".join(result)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Type something..."
    ),
    outputs=gr.Textbox(
        label="Top 5 Predictions"
    ),
    title="🔮 Next Word Predictor",
    theme=gr.themes.Soft()
)

demo.launch()
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# 1. LOAD MODEL & TOKENIZER
model = tf.keras.models.load_model('next_word_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# 2. PREDICT FUNCTION
def predict(text):
    tokens = tokenizer.texts_to_sequences([text.lower()])[0]
    tokens = pad_sequences([tokens], maxlen=5, padding='pre')
    probs  = model.predict(tokens, verbose=0)[0]
    top_ids = np.argsort(probs)[-5:][::-1]
    idx2word = {v: k for k, v in tokenizer.word_index.items()}
    return [(idx2word.get(i, '?'), float(probs[i])) for i in top_ids]

# 3. UI
st.title("🔮 Next Word Predictor")

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input:
        results = predict(user_input)
        st.subheader("Suggestions:")
        for word, prob in results:
            st.write(f"➡️ **{word}** — {prob:.2%}")
    else:
        st.warning("Please enter some text!")
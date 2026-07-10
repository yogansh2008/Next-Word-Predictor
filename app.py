import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")

st.title("🔮 Next Word Predictor")

# Load model and tokenizer (only once)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('next_word_model_new.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

# Predict function
def predict(text, top_n=5):
    tokens = tokenizer.texts_to_sequences([text.lower()])[0]
    tokens = pad_sequences([tokens], maxlen=5, padding='pre')
    probs = model.predict(tokens, verbose=0)[0]
    top_ids = np.argsort(probs)[-top_n:][::-1]
    idx2word = {v: k for k, v in tokenizer.word_index.items()}
    return [(idx2word.get(i, '?'), float(probs[i])) for i in top_ids]

# Keep track of the sentence being built
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

# Text input
user_input = st.text_input("Type your text here:", value=st.session_state.sentence)

# How many suggestions to show
top_n = st.slider("How many suggestions?", 1, 10, 5)

# Predict button
if st.button("🔮 Predict", type="primary"):
    st.session_state.sentence = user_input

# Show suggestions if there is text
if st.session_state.sentence.strip():
    results = predict(st.session_state.sentence, top_n)

    st.subheader("Click a word to add it to your sentence:")

    # Show each word as a button
    cols = st.columns(len(results))
    for i, (word, prob) in enumerate(results):
        with cols[i]:
            if st.button(f"{word}\n{prob:.0%}", key=f"word_{i}"):
                st.session_state.sentence = st.session_state.sentence + " " + word
                st.rerun()

    st.divider()

    # Show current sentence
    st.markdown(f"**Your sentence:** {st.session_state.sentence}")

    # Undo and Clear buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ Undo last word"):
            words = st.session_state.sentence.strip().split()
            st.session_state.sentence = " ".join(words[:-1])
            st.rerun()
    with col2:
        if st.button("🗑️ Clear all"):
            st.session_state.sentence = ""
            st.rerun()

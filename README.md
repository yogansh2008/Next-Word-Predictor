# Next-Word-Predictor
# 📝 Next Word Predictor using LSTM

A deep learning-based **Next Word Prediction** model built using **TensorFlow/Keras** and trained on the **TinyStories** dataset. Given an input sequence of words, the model predicts the most likely next word in the sentence.

---

## 🚀 Project Overview

This project demonstrates how Recurrent Neural Networks (LSTM) can be used for natural language processing tasks such as next-word prediction.

The model was trained on the **first 10,000 rows** of the **TinyStories** dataset to keep training lightweight while still learning meaningful language patterns.

---

## 📂 Dataset

* **Dataset:** TinyStories
* **Training Data Used:** First **10,000** rows
* **Type:** Simple English stories
* **Purpose:** Language Modeling / Next Word Prediction

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pickle
* Streamlit
* Google Colab

---

## 🧠 Model Architecture

* Text Tokenization
* Sequence Generation
* Padding Sequences
* Embedding Layer
* LSTM Layer
* Dense Output Layer (Softmax)

---

## 📊 Training Pipeline

1. Load TinyStories dataset.
2. Use the first 10,000 rows.
3. Clean and tokenize the text.
4. Generate input-output training sequences.
5. Pad all sequences to the same length.
6. Train the LSTM model.
7. Save:

   * Trained model (`.keras`)
   * Tokenizer (`tokenizer.pkl`)

---

## 📁 Project Structure

```text
Next-Word-Predictor/
│
├── app.py                     # Streamlit application
├── train_model.ipynb          # Model training notebook
├── tokenizer.pkl              # Saved tokenizer
├── next_word_model.keras      # Trained model
├── requirements.txt           # Required libraries
├── README.md
└── dataset/
```

---

## ▶️ Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Next-Word-Predictor.git
```

### 2. Navigate to the project

```bash
cd Next-Word-Predictor
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 💡 Example

**Input**

```text
Once upon a
```

**Prediction**

```text
delighted
```

---

## 📈 Future Improvements

* Train on the complete TinyStories dataset.
* Replace LSTM with a Transformer-based architecture.
* Predict the Top-5 next words instead of only one.
* Improve text preprocessing.
* Deploy the model with a public web interface.
* Add beam search for better sentence generation.

---

## 📌 Learning Outcomes

This project helped in understanding:

* Text preprocessing
* Tokenization
* Sequence generation
* Padding sequences
* Embedding layers
* LSTM networks
* Softmax classification
* Sparse Categorical Crossentropy
* Streamlit deployment
* Saving and loading deep learning models

---

## 🙌 Acknowledgements

* TinyStories Dataset
* TensorFlow
* Keras
* Streamlit
* Google Colab

---

## 📜 License

This project is intended for educational and learning purposes.
<img width="1803" height="868" alt="image" src="https://github.com/user-attachments/assets/aa0b0993-2881-417a-81fe-11ea1488eb11" />

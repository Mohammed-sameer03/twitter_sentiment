import streamlit as st
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import numpy as np

# ---------------------------
# 1. Load fine-tuned model
# ---------------------------
MODEL_PATH = "models/transformer_sentiment_model"
MODEL_NAME = "distilbert-base-uncased"

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# ---------------------------
# 2. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="centered")

st.title("🧠 Twitter Sentiment Analyzer (DistilBERT)")
st.write("Fine-tuned BERT model for detecting **Positive** or **Negative** sentiment.")

# Input box
user_input = st.text_area("Enter a tweet or sentence:", height=150, placeholder="Type something like: I love this product!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # ---------------------------
        # 3. Preprocess Input
        # ---------------------------
        inputs = tokenizer(
            [user_input],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="tf"
        )

        # ---------------------------
        # 4. Model Prediction
        # ---------------------------
        outputs = model(inputs)
        logits = outputs.logits
        predicted_class = int(tf.argmax(logits, axis=1).numpy()[0])

        label_map = {0: "Negative 😞", 1: "Positive 😊"}
        sentiment = label_map[predicted_class]

        # ---------------------------
        # 5. Display Result
        # ---------------------------
        st.success(f"**Predicted Sentiment:** {sentiment}")

        st.subheader("🔍 Confidence Scores")
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]
        st.write(f"Positive: {probs[1]*100:.2f}% | Negative: {probs[0]*100:.2f}%")

st.markdown("---")
st.markdown("Model fine-tuned using DistilBERT + TensorFlow")


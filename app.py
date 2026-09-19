import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 End-to-End Sentiment Analyzer")
st.write("Compare predictions across different NLP paradigms trained on consumer reviews.")

# Load models and artifacts with caching for fast inference
@st.cache_resource
def load_artifacts():
    bert_clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    with open("logistic_regression_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("bow_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
        
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
        
    nn = tf.keras.models.load_model("sentiment_dense_embedding.keras")
    
    return bert_clf, lr, vec, tok, nn

try:
    bert_clf, lr_model, bow_vec, tokenizer, dense_model = load_artifacts()
    st.success("All models & tokenizers loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")

# User Input
user_input = st.text_area(
    "Enter a review or sentence:", 
    "The plot was rather slow, but the cinematography was breathtaking."
)

selected_model = st.selectbox(
    "Choose Model Architecture:",
    [
        "DistilBERT (Transformer - 90.91% Acc)",
        "Dense Embedding + GlobalPooling (84.18% Acc)",
        "Logistic Regression (BOW - 80.91% Acc)"
    ]
)

if st.button("Analyze Sentiment", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        if "DistilBERT" in selected_model:
            res = bert_clf(user_input)[0]
            label = res['label']
            confidence = res['score'] * 100
        elif "Logistic Regression" in selected_model:
            features = bow_vec.transform([user_input])
            prob = lr_model.predict_proba(features)[0]
            pred = lr_model.predict(features)[0]
            label = "POSITIVE" if pred == 1 else "NEGATIVE"
            confidence = np.max(prob) * 100
        else:
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
            prob = dense_model.predict(padded, verbose=0)[0][0]
            label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
            confidence = (prob if prob >= 0.5 else (1 - prob)) * 100

        st.markdown("---")
        if label.upper() == "POSITIVE":
            st.success(f"### Sentiment: **{label.upper()}** 🎉")
        else:
            st.error(f"### Sentiment: **{label.upper()}** ⚠️")
            
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

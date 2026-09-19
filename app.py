import streamlit as st
import pickle
import numpy as np
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="NLP Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 End-to-End Sentiment Analyzer")
st.write("Compare predictions across traditional ML and Transformer architectures.")

# Load models with caching
@st.cache_resource
def load_artifacts():
    # 1. DistilBERT Pipeline (PyTorch backend)
    bert_clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # 2. Traditional Logistic Regression Baseline
    with open("logistic_regression_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("bow_vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
        
    return bert_clf, lr, vec

try:
    bert_clf, lr_model, bow_vec = load_artifacts()
    st.success("Models loaded successfully!")
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
        "DistilBERT (Transformer - 90.91% Test Acc)",
        "Logistic Regression (BOW - 80.91% Test Acc)"
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
        else:
            features = bow_vec.transform([user_input])
            prob = lr_model.predict_proba(features)[0]
            pred = lr_model.predict(features)[0]
            label = "POSITIVE" if pred == 1 else "NEGATIVE"
            confidence = np.max(prob) * 100

        st.markdown("---")
        if label.upper() == "POSITIVE":
            st.success(f"### Sentiment: **{label.upper()}** 🎉")
        else:
            st.error(f"### Sentiment: **{label.upper()}** ⚠️")
            
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

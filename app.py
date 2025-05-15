# app.py

import streamlit as st
import pickle
import re
import os
from datetime import datetime

# ===== Clean Text Function =====
def clean_arabic(text):
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'[\W_]+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    return text.strip()

# ===== Load Vectorizer =====
@st.cache_resource
def load_vectorizer():
    with open("vectorizer.pkl", "rb") as f:
        return pickle.load(f)

# ===== Load Selected Model =====
@st.cache_resource
def load_model(model_name):
    try:
        with open(f"model_{model_name}.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file for {model_name} not found.")
        return None

# ===== Streamlit UI =====
st.set_page_config(page_title="Arabic Complaint Classifier", layout="centered")
st.title("📋 Arabic Complaint Classifier")
st.markdown("Classify Arabic complaints using your trained machine learning models.")

with st.sidebar:
    st.header("Model Information")
    for model in ["LogisticRegression", "NaiveBayes", "SVM", "RandomForest"]:
        if os.path.exists(f"model_{model}.pkl"):
            mod_time = datetime.fromtimestamp(os.path.getmtime(f"model_{model}.pkl"))
            st.info(f"**{model}**: Trained on {mod_time.strftime('%Y-%m-%d')}")

user_input = st.text_area("✍️ Enter your Arabic complaint:", height=150)

model_name = st.selectbox(
    "Choose a model:",
    ["LogisticRegression", "NaiveBayes", "SVM", "RandomForest"]
)

if st.button("📝 Load Sample Text"):
    sample_texts = [
        "لم يتم تسليم المنتج في الوقت المحدد",
        "المنتج معيب ولا يعمل بشكل صحيح",
        "الخدمة العملاء سيئة للغاية"
    ]
    import random
    user_input = random.choice(sample_texts)
    st.session_state.user_input = user_input

if st.button("🔍 Classify"):
    if not user_input.strip():
        st.warning("Please enter some Arabic text.")
    else:
        clean_text = clean_arabic(user_input)
        vectorizer = load_vectorizer()
        model = load_model(model_name)
        X_input = vectorizer.transform([clean_text])
        prediction = model.predict(X_input)[0]
        st.success(f"📌 Predicted Class: **{prediction}**")

if st.button("🔄 Reset"):
    st.session_state.user_input = ""
    st.experimental_rerun()

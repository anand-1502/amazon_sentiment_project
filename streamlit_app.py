# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import nltk
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from wordcloud import WordCloud
import os
import joblib

# -------------------- Set Background --------------------
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
            }}
            .stApp::before {{
                content: "";
                position: absolute;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background-image: inherit;
                background-size: cover;
                background-position: center;
                filter: blur(3px);
                z-index: -1;
            }}
            .block-container {{
                background-color: rgba(255, 255, 255, 0.85);
                padding: 2rem;
                border-radius: 10px;
            }}
        </style>
    """, unsafe_allow_html=True)

set_background("background.jpg")

# -------------------- Load Models --------------------
nltk.download('vader_lexicon')

# Load Normal VADER
normal_sia = SentimentIntensityAnalyzer()
lexicon_path_normal = "models/vader_finetuned/custom_vader_lexicon.txt"
if os.path.exists(lexicon_path_normal):
    with open(lexicon_path_normal) as f:
        normal_lexicon = {word: float(score) for word, score in (line.strip().split('\t') for line in f)}
    normal_sia.lexicon.update(normal_lexicon)

# Load Boosted VADER
boosted_sia = SentimentIntensityAnalyzer()
lexicon_path_boosted = "models/vader_boosted/custom_vader_boosted_lexicon.txt"
if os.path.exists(lexicon_path_boosted):
    with open(lexicon_path_boosted) as f:
        boosted_lexicon = {word: float(score) for word, score in (line.strip().split('\t') for line in f)}
    boosted_sia.lexicon.update(boosted_lexicon)

# Load Boosted Hybrid Model (XGBoost)
boosted_hybrid_model = None
if os.path.exists('models/vader_boosted/vader_hybrid_model.joblib'):
    boosted_hybrid_model = joblib.load('models/vader_boosted/vader_hybrid_model.joblib')

# Load RoBERTa
roberta_tokenizer = AutoTokenizer.from_pretrained("models/roberta_finetuned/")
roberta_model = AutoModelForSequenceClassification.from_pretrained("models/roberta_finetuned/")

# Load Accuracies
roberta_accuracy = None
vader_accuracy = None
vader_boosted_accuracy = None

if os.path.exists("models/roberta_finetuned/accuracy.txt"):
    with open("models/roberta_finetuned/accuracy.txt", "r") as f:
        roberta_accuracy = float(f.read().strip())

if os.path.exists("models/vader_finetuned/accuracy.txt"):
    with open("models/vader_finetuned/accuracy.txt", "r") as f:
        vader_accuracy = float(f.read().strip())

if os.path.exists("models/vader_boosted/accuracy.txt"):
    with open("models/vader_boosted/accuracy.txt", "r") as f:
        vader_boosted_accuracy = float(f.read().strip())

# -------------------- Functions --------------------
def predict_vader(text, selected_sia, boosted=False):
    scores = selected_sia.polarity_scores(text)
    
    if boosted and boosted_hybrid_model is not None:
        features = np.array([[scores['neg'], scores['neu'], scores['pos'], scores['compound']]])
        prediction = boosted_hybrid_model.predict_proba(features)[0]
        labels = ['Negative', 'Neutral', 'Positive']
        score_dict = {labels[i]: prediction[i] for i in range(3)}
        return score_dict
    else:
        return {
            "Negative": scores['neg'],
            "Neutral": scores['neu'],
            "Positive": scores['pos']
        }

def predict_roberta(text):
    encoded_text = roberta_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = roberta_model(**encoded_text)
    probs = softmax(output.logits[0].numpy())
    labels = ['Negative', 'Neutral', 'Positive']
    score_dict = {labels[i]: float(probs[i]) for i in range(3)}
    return score_dict

def plot_pie(scores_dict):
    labels = ["Positive", "Neutral", "Negative"]
    sizes = [scores_dict[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda p: f'{p:.1f}%' if p > 1 else '',
        startangle=90,
        pctdistance=0.85,
        colors=["#66c2a5", "#fc8d62", "#8da0cb"],
        textprops=dict(color="black", fontsize=10, weight='bold')
    )
    ax.axis('equal')
    
    ax.legend(
        wedges, labels,
        title="Sentiment",
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
        fontsize=10,
        title_fontsize='13'
    )
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(centre_circle)

    return fig

def generate_wordcloud(words, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.subheader(title)
    st.pyplot(plt)

# -------------------- Streamlit Layout --------------------
st.title("📊 Fine-tuned Sentiment Analysis (VADER + Boosted VADER + RoBERTa)")
st.markdown("Analyze sentiments using **fine-tuned VADER**, **boosted VADER (Hybrid)**, and **fine-tuned RoBERTa** models on Amazon Reviews.")

user_input = st.text_area("💬 Enter your review:", "This product is absolutely amazing! Highly recommended.")

# Dropdown for model selection
model_choice = st.selectbox("🛠️ Choose VADER model to use:", ("Normal VADER", "Boosted VADER (Hybrid)"))

if model_choice == "Normal VADER":
    selected_sia = normal_sia
    boosted_flag = False
else:
    selected_sia = boosted_sia
    boosted_flag = True

if st.button("🔍 Analyze Sentiment"):
    vader_scores = predict_vader(user_input, selected_sia, boosted=boosted_flag)
    roberta_scores = predict_roberta(user_input)

    st.header("🔵 Sentiment Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📝 {model_choice} Prediction")
        st.success(f"Predicted: **{max(vader_scores, key=vader_scores.get)}**")
        st.write("Confidence Scores (%):")
        for label in ["Positive", "Neutral", "Negative"]:
            st.write(f"**{label}: {vader_scores[label]*100:.2f}%**")
        st.pyplot(plot_pie(vader_scores))

    with col2:
        st.subheader("🤖 RoBERTa Model Prediction")
        st.success(f"Predicted: **{max(roberta_scores, key=roberta_scores.get)}**")
        st.write("Confidence Scores (%):")
        for label in ["Positive", "Neutral", "Negative"]:
            st.write(f"**{label}: {roberta_scores[label]*100:.2f}%**")
        st.pyplot(plot_pie(roberta_scores))

    st.write("---")
    st.header("🖼️ WordCloud Visualization")
    words = user_input.lower().split()
    generate_wordcloud(words, "Combined WordCloud from Input Text")

    # -------------------- Model Accuracy Section --------------------
    st.write("---")
    st.header("📊 Overall Model Accuracy")

    col1, col2, col3 = st.columns(3)

    with col1:
        if vader_accuracy is not None:
            st.subheader("📝 Normal VADER Accuracy")
            st.info(f"**Accuracy: {vader_accuracy*100:.2f}%**")
        else:
            st.warning("Normal VADER accuracy not found.")

    with col2:
        if vader_boosted_accuracy is not None:
            st.subheader("⚡ Boosted VADER (Hybrid) Accuracy")
            st.info(f"**Accuracy: {vader_boosted_accuracy*100:.2f}%**")
        else:
            st.warning("Boosted VADER accuracy not found.")

    with col3:
        if roberta_accuracy is not None:
            st.subheader("🤖 RoBERTa Model Accuracy")
            st.info(f"**Accuracy: {roberta_accuracy*100:.2f}%**")
        else:
            st.warning("RoBERTa accuracy not found.")

    st.success("✅ Models were fine-tuned and evaluated on Amazon Fine Food Reviews Dataset!")

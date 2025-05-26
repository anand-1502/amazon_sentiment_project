# 🛍️ Amazon Sentiment Analyzer — VADER vs RoBERTa

Welcome to the **Amazon Sentiment Analyzer**, an interactive NLP-based application built using **RoBERTa**, **VADER**, and a hybrid **VADER + XGBoost** model. This project fine-tunes cutting-edge models on Amazon Fine Food Reviews and deploys a gorgeous **Streamlit** UI that makes sentiment comparison seamless and insightful.

---
amazon_sentiment_project/
│
├── data/                       # Raw data (Amazon reviews)
│
├── models/                    # Saved models
│   ├── roberta_finetuned/
│   └── vader_finetuned/
│
├── plots/                     # Training visualizations
│
├── streamlit_app.py           # Streamlit frontend
├── fine_tune_roberta.py       # RoBERTa training
├── fine_tune_vader.py         # VADER enhancement
├── project.ipynb              # EDA notebook
├── requirements.txt

---

## 🎯 Project Highlights

- 🔍 **EDA & Text Exploration** using `pandas`, `matplotlib`, `wordcloud`, `nltk`
- 🤖 **Modeling with:**
  - `VADER` (Rule-based)
  - `VADER + XGBoost` (Lexicon + ML hybrid)
  - `RoBERTa` (Transformer-based)
- 🧠 **Fine-Tuning** on 3-class sentiment (Negative, Neutral, Positive)
- 📊 Real-time **interactive visualizations** (Pie Charts, WordClouds, Accuracy Bars)
- 💻 Sleek **Streamlit app** with blurred background and custom styling
- 🧾 LIME integration for model explainability and confidence reporting

---


## 🔄 Project Workflow

### 📁 1. Data Loading & Labeling
- Load the dataset: `amazon_reviews.csv`
- Map review scores into sentiment classes:
  - **Negative**
  - **Neutral**
  - **Positive**

### 📊 2. Exploratory Data Analysis (EDA)
- Analyze text data to understand word usage
- Visualize class distribution and frequent terms

### 🤖 3. Model Development
- **RoBERTa Fine-Tuning**:
  - Train transformer model on labeled review texts
  - Save model checkpoints and training metrics
- **VADER Enhancement**:
  - Expand lexicon with dataset-specific terms
  - Evaluate and compare performance
- **Boosted VADER**:
  - Use VADER scores (`neg`, `neu`, `pos`, `compound`) as features
  - Train an XGBoost classifier for improved accuracy

### 🌐 4. Streamlit Dashboard
- Load all trained models into the web interface
- Accept user input for sentiment prediction
- Show visual outputs:
  - Predictions from all models
  - WordClouds and Pie Charts
  - Accuracy Comparison Bar Chart
  


## 🧪 Models Compared

| Model                | Type          | Accuracy | Notes                             |
|---------------------|---------------|----------|------------------------------------|
| VADER               | Rule-based    | Moderate | Quick, interpretable               |
| VADER + XGBoost     | Hybrid        | Improved | Boosted via custom lexicon + ML    |
| RoBERTa (fine-tuned)| Transformer   | High     | Context-aware, best accuracy       |

---

## 📸 Streamlit App Preview

### 🔹 App Landing Page  
> _(Insert Streamlit UI Screenshot Here)_

📌 **Placeholder #1 – First page of Streamlit**

---

### 🔹 Output Sample – Model Comparison  
> _(Insert prediction output showing VADER, Boosted VADER, RoBERTa)_

📌 **Placeholder #2 – First output screenshot**

---

### 🔹 Output Sample – WordClouds & Pie Charts  
> _(Insert visual output with WordCloud and Pie chart)_

📌 **Placeholder #3 – Second output screenshot**

---

## ⚙️ How to Set Up

> _Recommended: Use `conda` or `venv` for a clean environment._

```bash
# Step 0: Create environment
conda create -n amazon_sentiment python=3.10
conda activate amazon_sentiment

# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Fine-tune RoBERTa
python fine_tune_roberta.py

# Step 3: Expand & fine-tune VADER
python fine_tune_vader.py

# Step 4: Launch Streamlit App 🚀
streamlit run streamlit_app.py


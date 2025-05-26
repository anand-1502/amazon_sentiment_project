# 🛍️ Amazon Sentiment Analyzer — VADER vs RoBERTa

Welcome to the **Amazon Sentiment Analyzer**, an interactive NLP-based application built using **RoBERTa**, **VADER**, and a hybrid **VADER + XGBoost** model. This project fine-tunes cutting-edge models on Amazon Fine Food Reviews and deploys a gorgeous **Streamlit** UI that makes sentiment comparison seamless and insightful.

---
## 📁 Project Structure

| Folder/File                              | Description                                                                 |
|------------------------------------------|-----------------------------------------------------------------------------|
| `.ipynb_checkpoints/`                    | Auto-generated notebook checkpoints                                        |
| `EDA.ipynb`                              | Exploratory Data Analysis notebook                                         |
| `EDA copy.ipynb`                         | Backup of EDA notebook                                                     |
| `data/amazon_reviews.csv`               | Original Amazon Fine Food Reviews dataset                                 |
| `models/roberta_finetuned/`             | Fine-tuned RoBERTa model, tokenizer, and config files                     |
| ├── `accuracy.txt`                      | Text log of RoBERTa evaluation accuracy                                   |
| ├── `model.safetensors`                 | Trained RoBERTa model weights                                              |
| ├── `tokenizer_config.json`             | Tokenizer configuration                                                    |
| └── `vocab.json`, `merges.txt`, etc.    | Tokenizer components                                                       |
| `models/vader_boosted/`                 | Hybrid VADER + XGBoost model and custom lexicon                           |
| ├── `accuracy.txt`                      | Boosted model evaluation log                                               |
| ├── `custom_vader_boosted_lexicon.txt` | Extended VADER lexicon with domain-specific terms                         |
| └── `vader_hybrid_model.joblib`         | Trained XGBoost classifier on VADER scores                                |
| `models/vader_finetuned/`               | Custom fine-tuned VADER model output                                      |
| `outputs/`                              | Directory for future exported results or logs                             |
| `plots/accuracy_vs_steps.png`          | Accuracy vs steps graph for RoBERTa                                       |
| `plots/loss_vs_steps.png`              | Loss vs steps graph for RoBERTa                                           |
| `roberta model proof pdf`              | Evidence file for model verification                                      |
| `background.jpg`                        | Background image used in Streamlit UI                                     |
| `file explanations.rtf`                | Brief explanations of each script/module                                  |
| `final image.pdf`                       | Final visual or diagram for presentation                                  |
| `fine_tune_roberta.py`                 | Script to fine-tune RoBERTa on the dataset                                |
| `fine_tune_vader.py`                   | Script to enhance and evaluate VADER                                      |
| `fine_tune_vader_boosted.py`           | Script to train XGBoost on VADER scores                                   |
| `FSE 570 - ASME report.pdf`            | Final report document                                                      |
| `FSE 570 Final Presentation.pdf`       | Project presentation slides (PDF)                                         |
| `FSE 570 Final Presentation.pptx`      | Project presentation slides (PPTX)                                        |
| `how to setup.rtf`                     | Full setup guide for environment and scripts                              |
| `requirements.txt`                     | Python dependency list                                                     |
| `reviews.csv`                          | Another copy or version of the reviews dataset                            |
| `streamlit_app.py`                     | Streamlit web app to compare model predictions                            |
| `workflow file.rtf`                    | Workflow explanation document                                              |

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


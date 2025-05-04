# fine_tune_vader_boosted.py

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import xgboost as xgb

# -------------------- Step 1: Load Dataset --------------------
print("🔵 Loading dataset...")
df = pd.read_csv('data/amazon_reviews.csv')
df = df[['Text', 'Score']].rename(columns={'Text': 'text', 'Score': 'score'})

def map_sentiment(score):
    if score <= 2:
        return 0
    elif score == 3:
        return 1
    else:
        return 2

df['label'] = df['score'].apply(map_sentiment)

print(df.head())

# -------------------- Step 2: Initialize VADER --------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# -------------------- Step 3: Expand Lexicon  --------------------
print("🟠 Expanding and boosting VADER lexicon...")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip().split()

positive_texts = df[df['label'] == 2]['text'].tolist()
negative_texts = df[df['label'] == 0]['text'].tolist()

positive_words = Counter()
negative_words = Counter()

for text in positive_texts:
    positive_words.update(clean_text(text))

for text in negative_texts:
    negative_words.update(clean_text(text))

# Filter
excluded_words = ["small", "less", "more", "another", "only", "some", "few", "still", "might", "could"]

new_words = {}

for word, freq in positive_words.most_common(300):
    if word not in excluded_words:
        if freq > 100:
            new_words[word] = 4.0
        elif freq > 50:
            new_words[word] = 3.0
        else:
            new_words[word] = 2.0

for word, freq in negative_words.most_common(300):
    if word not in excluded_words and word not in new_words:
        if freq > 100:
            new_words[word] = -4.0
        elif freq > 50:
            new_words[word] = -3.0
        else:
            new_words[word] = -2.0

# Manually add strong phrases
special_phrases = {
    "highly recommend": 4.5,
    "would buy again": 4.0,
    "love this": 4.5,
    "worst ever": -4.5,
    "not worth it": -4.0,
    "do not recommend": -4.5,
    "never again": -4.5,
    "absolutely loved": 4.5,
    "absolutely terrible": -4.5,
    "horrible experience": -4.5,
    "fantastic product": 4.5,
    "really bad": -4.0,
    "extremely good": 4.0,
    "extremely bad": -4.0,
    "utterly disappointing": -4.5
}

sia.lexicon.update(new_words)
sia.lexicon.update(special_phrases)

# -------------------- Step 4: Save Boosted Lexicon --------------------
os.makedirs('models/vader_boosted/', exist_ok=True)
with open('models/vader_boosted/custom_vader_boosted_lexicon.txt', 'w') as f:
    for word, score in sia.lexicon.items():
        f.write(f"{word}\t{score}\n")

print("✅ Boosted VADER lexicon saved!")

# -------------------- Step 5: Prepare Features for Hybrid Model --------------------
print("🔵 Preparing VADER score features for hybrid model...")

# Extract features
features = []
for text in df['text']:
    scores = sia.polarity_scores(text)
    features.append([scores['neg'], scores['neu'], scores['pos'], scores['compound']])

features = pd.DataFrame(features, columns=['neg', 'neu', 'pos', 'compound'])
labels = df['label']

# -------------------- Step 6: Train XGBoost Model --------------------
print("🛠️ Training XGBoost on VADER features...")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------- Step 7: Evaluate Model --------------------
y_pred = model.predict(X_test)
hybrid_accuracy = accuracy_score(y_test, y_pred)

print(f"📈 Hybrid Boosted VADER (XGBoost) Accuracy: {hybrid_accuracy*100:.2f}%")

# -------------------- Step 8: Save Everything --------------------
joblib.dump(model, 'models/vader_boosted/vader_hybrid_model.joblib')
with open('models/vader_boosted/accuracy.txt', 'w') as f:
    f.write(str(hybrid_accuracy))

print("✅ Saved Hybrid VADER (XGBoost) model and updated accuracy.")

# fine_tune_vader.py

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from collections import Counter
import re

# Step 1: Load Dataset
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

# df = df.sample(n=10000, random_state=42).reset_index(drop=True)
df['label'] = df['score'].apply(map_sentiment)

print(df.head())

# Step 2: Initialize VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Step 3: Expand Lexicon
print("🟠 Auto-generating VADER lexicon...")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    return text.split()

positive_texts = df[df['label'] == 2]['text'].tolist()
negative_texts = df[df['label'] == 0]['text'].tolist()

positive_words = Counter()
negative_words = Counter()

for text in positive_texts:
    positive_words.update(clean_text(text))

for text in negative_texts:
    negative_words.update(clean_text(text))

top_positive = dict(positive_words.most_common(100))
top_negative = dict(negative_words.most_common(100))

new_words = {word: 2.0 for word in top_positive}
for word in top_negative:
    if word not in new_words:
        new_words[word] = -2.0

sia.lexicon.update(new_words)

# Save expanded lexicon
os.makedirs('models/vader_finetuned/', exist_ok=True)
with open('models/vader_finetuned/custom_vader_lexicon.txt', 'w') as f:
    for word, score in sia.lexicon.items():
        f.write(f"{word}\t{score}\n")

print("✅ Custom VADER lexicon saved!")

# Step 4: Evaluate
print("🟢 Evaluating expanded VADER...")

def vader_sentiment(text):
    score = sia.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 2
    elif compound <= -0.05:
        return 0
    else:
        return 1

df['vader_prediction'] = df['text'].apply(vader_sentiment)

# Save Accuracy
accuracy = (df['label'] == df['vader_prediction']).mean()
print(f"📈 VADER Accuracy: {accuracy*100:.2f}%")

with open('models/vader_finetuned/accuracy.txt', 'w') as f:
    f.write(str(accuracy))

print("✅ Saved accuracy and model ")

# fine_tune_roberta.py

import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

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

df['label'] = df['score'].apply(map_sentiment)

# Sample 100 for fast training
df = df.sample(n=100, random_state=42).reset_index(drop=True)
print(df.head())

# Step 2: Preprocessing
print("🟠 Splitting dataset...")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels}).map(tokenize, batched=True)
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels}).map(tokenize, batched=True)

# Step 3: Model Setup
print("🟡 Setting up model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment", num_labels=3
)

# Step 4: Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    logging_steps=25,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Step 5: Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Step 6: Train
print("🟢 Training started...")
trainer.train()
print("✅ Training finished!")

# Step 7: Save
os.makedirs("models/roberta_finetuned", exist_ok=True)
model.save_pretrained("models/roberta_finetuned/")
tokenizer.save_pretrained("models/roberta_finetuned/")

# Save Accuracy
accuracy = trainer.evaluate()['eval_accuracy']
with open('models/roberta_finetuned/accuracy.txt', 'w') as f:
    f.write(str(accuracy))

print("✅ Saved model and accuracy ")

# Step 8: Plot Training Curves
os.makedirs("plots", exist_ok=True)

metrics = trainer.state.log_history
train_loss = [x['loss'] for x in metrics if 'loss' in x]
eval_accuracy = [x['eval_accuracy'] for x in metrics if 'eval_accuracy' in x]
steps_loss = [x['step'] for x in metrics if 'loss' in x]
steps_acc = [x['step'] for x in metrics if 'eval_accuracy' in x]

# Loss plot
plt.figure(figsize=(8,5))
plt.plot(steps_loss, train_loss)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Loss vs Steps')
plt.grid()
plt.savefig('plots/loss_vs_steps.png')
plt.close()

# Accuracy plot
plt.figure(figsize=(8,5))
plt.plot(steps_acc, eval_accuracy)
plt.xlabel('Steps')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs Steps')
plt.grid()
plt.savefig('plots/accuracy_vs_steps.png')
plt.close()

print("✅ Saved training plots.")

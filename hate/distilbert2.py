
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_fscore_support
from datetime import datetime
import csv
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

def prepare_hate_data(path):
    df = pd.read_excel(path)
    df = df[['Translated Post Description', 'Hate']].dropna()
    df['Translated Post Description'] = df['Translated Post Description'].apply(lambda x: " ".join(str(x).split()[:250]))
    df['Hate'] = df['Hate'].str.lower().str.strip()
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Hate'])
    X_train, X_test, y_train, y_test = train_test_split(
        df['Translated Post Description'], df['label'],
        test_size=0.2, stratify=df['label'], random_state=42
    )
    return X_train, X_test, y_train, y_test, le

data_path = "../MonkeyPox.xlsx"
X_train, X_test, y_train, y_test, le = prepare_hate_data(data_path)

train_ds = Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()})
test_ds = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = {
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }
    precision, recall, _, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    metrics["precision"] = precision
    metrics["recall"] = recall
    if num_labels == 2:
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    else:
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        metrics["roc_auc"] = roc_auc_score(labels, probs, multi_class="ovo", average="macro")
    return metrics

training_args = TrainingArguments(
    output_dir="./distilbert_hate_output_80_20_no_weights",
    logging_dir="./distilbert_hate_output_80_20_no_weights/logs",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    fp16=True,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    report_to="none",
    learning_rate=2e-5,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

model.save_pretrained("./distilbert_hate_final_model_80_20_no_weights")
tokenizer.save_pretrained("./distilbert_hate_final_model_80_20_no_weights")
print("✅ Model başarıyla 'distilbert_hate_final_model_80_20_no_weights' klasörüne kaydedildi (en iyi model).")

predictions = trainer.predict(test_ds)
y_pred = predictions.predictions.argmax(-1)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"Macro F1 Score: {f1_macro:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")

if num_labels == 2:
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    auc_score = roc_auc_score(y_test, probs)
else:
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    auc_score = roc_auc_score(y_test, probs, multi_class="ovo", average="macro")

print(f"ROC AUC Score: {auc_score:.4f}")

def save_metrics_to_csv(filepath, metrics_dict):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
metrics = {
    "timestamp": timestamp,
    "model": model_ckpt,
    "num_labels": num_labels,
    "f1_macro": round(f1_macro, 4),
    "f1_weighted": round(f1_weighted, 4),
    "roc_auc": round(auc_score, 4),
    "train_batch_size": training_args.per_device_train_batch_size,
    "eval_batch_size": training_args.per_device_eval_batch_size,
    "num_train_epochs": trainer.state.epoch,
    "early_stopping_patience": 3,
    "learning_rate": training_args.learning_rate
}

csv_path = "distilbert_hate_metrics_log_80_20_no_weights.csv"
save_metrics_to_csv(csv_path, metrics)
print(f"\n✅ Metrikler başarıyla '{csv_path}' dosyasına kaydedildi.")

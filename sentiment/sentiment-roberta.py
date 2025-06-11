import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from datetime import datetime
import csv
import os

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 📊 Veri hazırlama (sentiment)
def prepare_sentiment_data(path):
    df = pd.read_excel(path)
    df = df[['Translated Post Description', 'Sentiment']].dropna()
    df['Translated Post Description'] = df['Translated Post Description'].apply(lambda x: " ".join(str(x).split()[:250]))
    df['Sentiment'] = df['Sentiment'].str.lower().str.strip()

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['Translated Post Description'], df['label'],
        test_size=0.1, stratify=df['label'], random_state=42
    )

    return X_train, X_test, y_train, y_test, le

# 📂 Dosya yolu
data_path = "../MonkeyPox.xlsx"

# 📥 Veriyi yükle
X_train, X_test, y_train, y_test, le = prepare_sentiment_data(data_path)

# 🤖 Huggingface Dataset
train_ds = Dataset.from_dict({"text": X_train, "label": y_train})
test_ds = Dataset.from_dict({"text": X_test, "label": y_test})

# Tokenizer ve model
model_ckpt = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# Model
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./roberta_sentiment_output",
    logging_dir="./roberta_sentiment_output/logs",
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    fp16=True,
    save_strategy="no",
    report_to="none",
    load_best_model_at_end=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer
)

# Eğitimi başlat
trainer.train()

# Modeli kaydet
model.save_pretrained("./roberta_sentiment_final_model")
tokenizer.save_pretrained("./roberta_sentiment_final_model")
print("✅ Model başarıyla 'roberta_sentiment_final_model' klasörüne kaydedildi.")

# Değerlendirme
predictions = trainer.predict(test_ds)
y_pred = predictions.predictions.argmax(-1)

# METRİKLER
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")
print(f"Macro F1 Score: {f1_macro:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")

# ROC AUC
if num_labels == 2:
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
    auc_score = roc_auc_score(y_test, probs)
else:
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    auc_score = roc_auc_score(y_test, probs, multi_class="ovo", average="macro")

print(f"ROC AUC Score: {auc_score:.4f}")

# METRİKLERİ CSV'ye kaydet
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
    "num_train_epochs": training_args.num_train_epochs
}

csv_path = "roberta_sentiment_metrics_log.csv"
save_metrics_to_csv(csv_path, metrics)
print(f"\n✅ Metrikler başarıyla '{csv_path}' dosyasına kaydedildi.")

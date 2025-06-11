import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

# MODELİ YÜKLE
model_path = "./deberta_sentiment_final_model"  # sentiment modeli ise bunu kullan
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# VERİYİ YÜKLE
df = pd.read_excel("../MonkeyPox.xlsx")
df = df[['Translated Post Description', 'Sentiment']].dropna()
df['Translated Post Description'] = df['Translated Post Description'].apply(lambda x: " ".join(str(x).split()[:250]))
df['Sentiment'] = df['Sentiment'].str.lower().str.strip()

# LABEL ENCODE
le = LabelEncoder()
df['label'] = le.fit_transform(df['Sentiment'])

# TEST VERİSİ
X_train, X_test, y_train, y_test = train_test_split(
    df['Translated Post Description'],
    df['label'],
    test_size=0.1,
    stratify=df['label'],
    random_state=42
)

# TOKENİZASYON
test_ds = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})
test_ds = test_ds.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=512), batched=True)

# MODEL İLE TAHMİN
all_logits = []
model.eval()
with torch.no_grad():
    for i in range(0, len(test_ds), 16):
        batch = test_ds[i:i+16]
        inputs = {
            'input_ids': torch.tensor(batch['input_ids']).to(model.device),
            'attention_mask': torch.tensor(batch['attention_mask']).to(model.device)
        }
        outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu().numpy())

# SONUÇLARI HAZIRLA
all_logits = np.concatenate(all_logits, axis=0)
probs = softmax(torch.tensor(all_logits), dim=1).numpy()
num_labels = model.config.num_labels

# ROC AUC
if num_labels == 2:
    auc = roc_auc_score(y_test, probs[:, 1])
    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Binary")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    y_test_binarized = label_binarize(y_test, classes=np.arange(num_labels))
    auc = roc_auc_score(y_test_binarized, probs, multi_class="ovo", average="macro")

    plt.figure(figsize=(8, 6))
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"{le.classes_[i]} (AUC: {roc_auc_score(y_test_binarized[:, i], probs[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Multi-Class Sentiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(f"\n✅ ROC AUC Skoru (macro average): {auc:.4f}")

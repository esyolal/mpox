import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import matplotlib.pyplot as plt

# 🔧 MODELİ VE TOKENIZER'I YÜKLE
model_path = "./bert_stress_final_model"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 📊 VERİYİ HAZIRLA
df = pd.read_excel("../MonkeyPox.xlsx")
df = df[['Translated Post Description', 'Stress or Anxiety']].dropna()
df['Translated Post Description'] = df['Translated Post Description'].apply(lambda x: " ".join(str(x).split()[:250]))
df['Stress or Anxiety'] = df['Stress or Anxiety'].str.lower().str.strip()

# 🔁 LABEL ENCODER
le = LabelEncoder()
df['label'] = le.fit_transform(df['Stress or Anxiety'])

# 🧪 TEST VERİSİNİ AYIR
X_train, X_test, y_train, y_test = train_test_split(
    df['Translated Post Description'],
    df['label'],
    test_size=0.1,
    stratify=df['label'],
    random_state=42
)

# 🤖 TOKENIZE EDİLMİŞ TEST VERİSİNİ DÖNÜŞTÜR
test_ds = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})
test_ds = test_ds.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=512), batched=True)

# 🔮 MODEL İLE TAHMİN ET
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

# 🧩 LOGITS TO ROC AUC
all_logits = np.concatenate(all_logits, axis=0)

if model.config.num_labels == 2:
    probs = torch.nn.functional.softmax(torch.tensor(all_logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(y_test, probs)

    # ✅ ROC EĞRİSİ HESAPLA
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    # ✅ ROC EĞRİSİNİ ÇİZ
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Stress/Anxiety Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print("ROC eğrisi sadece binary classification (2 sınıf) için destekleniyor.")

print(f"\n✅ ROC AUC Skoru: {auc:.4f}")

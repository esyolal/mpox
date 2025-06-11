import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# 🔧 Kendi eğittiğin modeli ve tokenizer'ı yükle
model_path = "./deberta-v3-small_stress_final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 📊 Veri hazırlığı
df = pd.read_excel("../MonkeyPox.xlsx")
df = df[['Translated Post Description', 'Stress or Anxiety']].dropna()
df['Translated Post Description'] = df['Translated Post Description'].apply(lambda x: " ".join(str(x).split()[:250]))
df['Stress or Anxiety'] = df['Stress or Anxiety'].str.lower().str.strip()

# 🎯 Label encode
le = LabelEncoder()
df['label'] = le.fit_transform(df['Stress or Anxiety'])

# 🧪 Test setini ayır
X_train, X_test, y_train, y_test = train_test_split(
    df['Translated Post Description'],
    df['label'],
    test_size=0.1,
    stratify=df['label'],
    random_state=42
)

# 🔮 Model ile tahmin yap
all_logits = []
with torch.no_grad():
    for i in range(0, len(X_test), 16):
        batch_texts = X_test.iloc[i:i+16].tolist()
        inputs = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu().numpy())

# 🧩 ROC AUC hesapla
all_logits = np.concatenate(all_logits, axis=0)

if model.config.num_labels == 2:
    probs = torch.nn.functional.softmax(torch.tensor(all_logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(y_test, probs)

    # ✅ ROC eğrisi çiz
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fine-tuned DeBERTa v3 Small")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("ROC eğrisi sadece binary classification için çalışır.")

print(f"\n✅ ROC AUC Skoru: {auc:.4f}")

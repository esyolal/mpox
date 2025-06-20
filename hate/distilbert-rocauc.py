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
# Hate modelinizi yüklemek için yolu güncelledik
model_path = "./distilbert_hate_final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval() # Modeli değerlendirme moduna al

# VERİYİ YÜKLE
df = pd.read_excel("../MonkeyPox.xlsx")
# 'Sentiment' yerine 'Hate' sütununu kullanıyoruz
df = df[['Translated Post Description', 'Hate']].dropna()
df['Translated Post Description'] = df['Translated Post Description'].apply(lambda x: " ".join(str(x).split()[:250]))
df['Hate'] = df['Hate'].str.lower().str.strip()

# LABEL ENCODE
le = LabelEncoder()
df['label'] = le.fit_transform(df['Hate'])

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
model.eval() # Modeli değerlendirme moduna al
with torch.no_grad(): # Gradyan hesaplamalarını devre dışı bırak (bellek ve hız için)
    # Batch boyutunu modelin eğitiminde kullanılan batch boyutuyla uyumlu tutmak iyi bir uygulamadır.
    # Burada 16 olarak ayarlı, eğitimde 8 kullanıyorduk. 8 olarak değiştirebiliriz.
    batch_size_inference = 8 # Eğitimdeki batch boyutunu kullanmak daha tutarlı olabilir
    for i in range(0, len(test_ds), batch_size_inference):
        batch = test_ds[i:i+batch_size_inference]
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

# ROC AUC Eğrisi Çizimi
if num_labels == 2:
    # İkili sınıflandırma (hate ve not hate) için ROC AUC
    auc = roc_auc_score(y_test, probs[:, 1])
    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hate Classification (Binary)") # Başlığı güncelledik
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    # Çoklu sınıflandırma için ROC AUC (eğer 2'den fazla sınıf olsaydı)
    y_test_binarized = label_binarize(y_test, classes=np.arange(num_labels))
    auc = roc_auc_score(y_test_binarized, probs, multi_class="ovo", average="macro")

    plt.figure(figsize=(8, 6))
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], probs[:, i])
        # Sınıf isimlerini le.classes_ ile alıyoruz
        plt.plot(fpr, tpr, label=f"{le.classes_[i]} (AUC: {roc_auc_score(y_test_binarized[:, i], probs[:, i]):.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hate Classification (Multi-Class)") # Başlığı güncelledik
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(f"\n✅ ROC AUC Skoru (macro average): {auc:.4f}")

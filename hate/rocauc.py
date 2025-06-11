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

# 🔌 GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 📊 Veriyi yükle ve hazırla (Tüm modeller için bir kez)
def prepare_hate_data(path):
    df = pd.read_excel(path)
    df = df[['Translated Post Description', 'Hate']].dropna()
    df['Translated Post Description'] = df['Translated Post Description'].apply(lambda x: " ".join(str(x).split()[:250]))
    df['Hate'] = df['Hate'].str.lower().str.strip()

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['Hate'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['Translated Post Description'],
        df['label'],
        test_size=0.1,
        stratify=df['label'],
        random_state=42
    )
    return X_test, y_test, le

# 📂 Veri dosyası
data_path = "../MonkeyPox.xlsx" 

# 📥 Veriyi hazırla
X_test, y_test, le = prepare_hate_data(data_path)
num_labels = len(le.classes_) # Sınıf sayısını al

# Karşılaştırılacak modellerin listesi
# Model adları ve kaydedildikleri yollar
models_to_compare = [
    {"name": "BERT", "path": "./bert_hate_final_model"},
    {"name": "DistilBERT", "path": "./distilbert_hate_final_model"},
    {"name": "DeBERTa-v3-small", "path": "./deberta_v3_small_hate_final_model"},
    {"name": "RoBERTa", "path": "./roberta_hate_final_model"}
]

plt.figure(figsize=(10, 8)) # Daha büyük bir grafik boyutu

# Her bir model için ROC eğrisini çiz
for model_info in models_to_compare:
    model_name = model_info["name"]
    model_path = model_info["path"]
    
    print(f"\n--- {model_name} modelini yüklüyor ve tahminler yapıyor... ---")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = model.to(device)
        model.eval() # Modeli değerlendirme moduna al
    except Exception as e:
        print(f"Hata: {model_name} modeli yüklenemedi veya bulunamadı: {e}")
        print(f"Lütfen '{model_path}' yolunun doğru olduğundan ve modelin kaydedildiğinden emin olun.")
        continue # Sonraki modele geç

    # TOKENİZASYON (Her modelin kendi tokenizer'ı ile)
    test_ds = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})
    test_ds = test_ds.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=512), batched=True)

    # MODEL İLE TAHMİN
    all_logits = []
    batch_size_inference = 8 # Eğitimdeki batch boyutunu kullanmak daha tutarlı olabilir
    with torch.no_grad(): # Gradyan hesaplamalarını devre dışı bırak
        for i in range(0, len(test_ds), batch_size_inference):
            batch = test_ds[i:i+batch_size_inference]
            inputs = {
                'input_ids': torch.tensor(batch['input_ids']).to(model.device),
                'attention_mask': torch.tensor(batch['attention_mask']).to(model.device)
            }
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    probs = softmax(torch.tensor(all_logits), dim=1).numpy()

    # ROC AUC ve Eğri Hesaplama
    if num_labels == 2:
        # İkili sınıflandırma (hate ve not hate) için ROC AUC
        auc = roc_auc_score(y_test, probs[:, 1])
        fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})", linewidth=2)
    else:
        # Çoklu sınıflandırma için ROC AUC (eğer 2'den fazla sınıf olsaydı)
        y_test_binarized = label_binarize(y_test, classes=np.arange(num_labels))
        auc = roc_auc_score(y_test_binarized, probs, multi_class="ovo", average="macro")
        # Çoklu sınıflandırmada her sınıf için ayrı ROC eğrisi çizmek yerine
        # basitleştirilmiş bir yaklaşım veya belirli bir sınıfın eğrisi çizilebilir.
        # Bu örnekte, sadece genel AUC'yi gösterip, tek bir eğri çiziyoruz (örneğin ilk sınıf için).
        # Daha detaylı çoklu sınıf ROC için ayrı bir döngüye ihtiyaç duyulur.
        print(f"Uyarı: Çoklu sınıflandırma için tek bir ROC eğrisi çiziliyor. Detaylı analiz için her sınıf için ayrı eğri çizilmesi gerekebilir.")
        fpr, tpr, _ = roc_curve(y_test_binarized[:, 0], probs[:, 0]) # İlk sınıfın eğrisi
        plt.plot(fpr, tpr, label=f"{model_name} (Macro AUC = {auc:.4f})", linewidth=2)


# "Random Guess" (Rastgele Tahmin) Çizgisi
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess", alpha=0.7)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Hate Classification - ROC Curve Comparison Across Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n✅ Tüm modeller için ROC AUC eğrileri başarıyla çizildi.")

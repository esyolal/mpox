# 🧠 Transformer-Based Text Classification: Hate, Sentiment & Stress Detection

This repository presents a comparative study of **Transformer models** applied to three key text classification tasks on social media content:

- 🧨 **Hate Speech Detection**
- 💬 **Sentiment Analysis**
- 😰 **Stress/Anxiety Detection**

The dataset is derived from MonkeyPox-related social media posts and includes labeled examples for all three tasks.

---

## 🚀 Models Used

We fine-tuned and evaluated the following models for each classification task:

| Model                      | Parameters | Type                |
|---------------------------|------------|---------------------|
| `distilbert-base-uncased` | ~66M       | Lightweight BERT    |
| `bert-base-uncased`       | ~110M      | Standard BERT       |
| `roberta-base`            | ~125M      | Optimized BERT      |
| `microsoft/deberta-v3-small` | ~86M    | Disentangled BERT   |

All models were evaluated under identical conditions (same dataset splits, preprocessing steps, and training parameters).

---

## 🧪 Tasks & Pipeline

Each model was fine-tuned separately for:

- **Hate classification** (e.g., `hate`, `not hate`)
- **Sentiment analysis** (e.g., `anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`)
- **Stress/anxiety detection** (e.g., `stress`, `no stress`)

All training scripts follow the same structure:

1. ✅ Data preprocessing and label encoding
2. ✅ Tokenization using Hugging Face tokenizer
3. ✅ Fine-tuning with Hugging Face `Trainer`
4. ✅ Evaluation using `F1`, `ROC AUC`, `Precision`, `Recall`
5. ✅ Automatic saving of best models and metric logs

---

## 📂 Project Structure

```
.
├── data/
│   └── MonkeyPox.xlsx                     # Labeled dataset
├── models/
│   ├── distilbert_hate_final_model/
│   ├── bert_sentiment_final_model/
│   ├── roberta_stress_final_model/
│   └── deberta_hate_final_model/
├── scripts/
│   ├── train_distilbert.py
│   ├── train_bert.py
│   ├── train_roberta.py
│   └── train_deberta.py
├── results/
│   └── classification_metrics.csv         # All evaluation metrics
```

---

## 📊 Sample Evaluation Results

| Task      | Model                 | F1 Macro | ROC AUC | Best Epoch |
|-----------|------------------------|----------|---------|-------------|
| Hate      | DeBERTa-v3-small       | 0.94     | 0.96    | 3           |
| Sentiment | BERT Base              | 0.91     | 0.93    | 4           |
| Stress    | RoBERTa Base           | 0.89     | 0.92    | 5           |
| Hate      | DistilBERT             | 0.90     | 0.91    | 4           |

> 🗂 All metrics are automatically logged in `classification_metrics.csv`

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install transformers datasets scikit-learn torch openpyxl
```

---

## 🔮 Future Work

- [ ] Cross-task transfer learning  
- [ ] Ensemble modeling for better generalization  
- [ ] Deployable web interface (Gradio / Streamlit)  
- [ ] Public dataset release and benchmark comparison  

---

## 👨‍💻 Author

Developed by **Emre Sebati Yolal**  
🎓 MSc Candidate in Information Systems Engineering  
💡 Specializing in NLP, Deep Learning, and .NET backend development

---

## 🧾 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

# Experiment 7: Sentiment Analysis on IMDB – Logistic Regression vs. LSTM vs. Transformer vs. BERT

This folder contains a Jupyter Notebook that trains four sentiment classifiers on the **IMDB Dataset of 50K Movie Reviews** and compares them with **Precision, Recall, F1-score and Confusion Matrices**.

## Files

| File | Description |
|------|-------------|
| [`Experiment_7.ipynb`](Experiment_7.ipynb) | Notebook with all models and the comparative analysis (outputs from a Google Colab **NVIDIA L4 GPU** run on the full data) |
| `IMDB Dataset.csv` | Kaggle dataset (50,000 reviews; *not committed* – created automatically by the notebook from the Hugging Face mirror `stanfordnlp/imdb` if missing) |
| [`Experiment-7.docx`](Experiment-7.docx) | Problem statement |

**Dataset:** [Kaggle – IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) · reference code: [Ankit152/IMDB-sentiment-analysis](https://github.com/Ankit152/IMDB-sentiment-analysis)

---

## Pipeline

1. **Load** `IMDB Dataset.csv` (25,000 positive / 25,000 negative).
2. **Clean** reviews: HTML unescape, remove `<br />` and other tags, URLs and non-letters; lower-case.
3. **Split** 80/20 stratified → 40,000 train / 10,000 test (seed 42).
4. **Train and evaluate** four models; plot confusion matrices; compare.

## Models

| # | Model | Details |
|---|-------|---------|
| 1 | **Logistic Regression** | TF-IDF (50K uni+bi-gram features, sublinear TF, stop words removed) + `LogisticRegression(C=4)` |
| 2 | **Bidirectional LSTM** | Embedding 128 → 2-layer BiLSTM (128 units/direction, packed sequences) → Linear; 3 epochs, Adam 1e-3 |
| 3 | **Vanilla Transformer** | Embedding 128 + sinusoidal positional encoding → 2 encoder layers (4 heads, FF 256, padding mask) → masked mean pooling → Linear; 4 epochs, Adam 5e-4 |
| 4 | **BERT** | `bert-base-uncased` + classification head, fine-tuned 1 epoch (AdamW 2e-5, linear warm-up/decay, batch 16, max length 256) |

Neural models 2 and 3 use a 20,000-word vocabulary and sequences of 256 tokens.

---

## Results (full 10,000-review test set)

| Model | Accuracy | Precision | Recall | F1-score | FP | FN |
|-------|---------:|----------:|-------:|---------:|---:|---:|
| Logistic Regression | 0.9020 | 0.8954 | 0.9104 | 0.9028 | 532 | 448 |
| LSTM | 0.8504 | 0.8194 | 0.8990 | 0.8573 | 991 | 505 |
| Transformer | 0.8386 | 0.8103 | 0.8842 | 0.8456 | 1035 | 579 |
| **BERT** | **0.9252** | **0.9254** | **0.9250** | **0.9252** | **373** | **375** |

Training time on an L4 GPU: LR ≈ 25 s, LSTM ≈ 30 s/epoch, Transformer ≈ 10 s/epoch, BERT ≈ 14 min (1 epoch).

**Observations**
- **BERT** is best on every metric and has a balanced confusion matrix; it is the only model that correctly classifies *"The visuals were great but the story was not good at all"* as negative.
- **TF-IDF + Logistic Regression** is a very strong, cheap baseline (F1 0.903).
- The LSTM and Transformer trained from scratch are biased towards the positive class (high recall, lower precision) and stay below the TF-IDF baseline without pretraining.
- **Ranking (F1):** BERT > Logistic Regression > LSTM > Transformer.

---

## How to Run

**Google Colab / Kaggle (recommended):** upload the notebook, select a **GPU runtime** (Colab: *Runtime → Change runtime type → T4/L4/A100 GPU*) and *Run all*. The setup cell installs missing packages and the dataset is downloaded automatically.

**Locally (uv):** `uv sync` in the repository root, then run the notebook with the `.venv` kernel. On a CPU-only machine the notebook automatically fine-tunes BERT on a 2,000-review subset and evaluates all models on a common 1,000-review test subset (`BERT_TRAIN_SIZE`, `EVAL_SUBSET_SIZE`) to keep the run time manageable.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **TF-IDF** | Term frequency × inverse document frequency feature weighting |
| **LSTM** | Recurrent network with input/forget/output gates for long-range dependencies |
| **Self-attention** | `softmax(QKᵀ/√d)V` – every token attends to every other token |
| **BERT** | 12-layer Transformer encoder pretrained with masked language modelling, fine-tuned for classification |
| **Precision / Recall / F1** | TP/(TP+FP), TP/(TP+FN) and their harmonic mean |
| **Confusion matrix** | Table of TN, FP, FN, TP counts |

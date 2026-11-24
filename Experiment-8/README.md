# Experiment 8: Abstractive Text Summarization with mT5 and mBART (Hindi XL-Sum)

This folder contains a Jupyter Notebook that summarises **Hindi** news articles from the **XL-Sum** dataset with two multilingual Transformer sequence-to-sequence models, **mT5** and **mBART-50**, and compares them using **ROUGE-1, ROUGE-2, ROUGE-L** and **mBERT-based BERTScore**.

## Files

| File | Description |
|------|-------------|
| [`Experiment_8.ipynb`](Experiment_8.ipynb) | Notebook: data loading, summarization with both models, evaluation and comparison (executed, CPU run on 100 articles) |
| [`hindi_xlsum_generated_summaries.csv`](hindi_xlsum_generated_summaries.csv) | Article, reference and both generated summaries with per-article ROUGE-L / mBERTScore |
| [`Experiment-8.docx`](Experiment-8.docx) | Problem statement |

**Dataset:** [csebuetnlp/xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum) – Hindi subset (70,778 train / 8,847 validation / 8,847 test BBC article–summary pairs). It is downloaded automatically (Parquet files) on first run.

---

## Models

| Model | Hugging Face checkpoint | Base model | Params |
|-------|-------------------------|------------|-------:|
| **mT5** | `csebuetnlp/mT5_multilingual_XLSum` | `google/mt5-base`, fine-tuned on 45 XL-Sum languages | 582M |
| **mBART-50** | `mskayacioglu/mbart-xlsum-2` | `facebook/mbart-large-50-many-to-many-mmt`, fine-tuned on 15 XL-Sum languages | 611M |

**Decoding (identical for both):** input truncated to 512 tokens, `num_beams=4`, `max_new_tokens=84`, `no_repeat_ngram_size=2`, `length_penalty=1.0`. For mBART the source/target language is `hi_IN` and `forced_bos_token_id` forces the Hindi language token.

## Evaluation

- **ROUGE-1 / ROUGE-2 / ROUGE-L (F1)** with `rouge_score`, using a custom **Devanagari-aware tokenizer** (removes punctuation and the danda `।`, whitespace split, no stemming). The default `rouge_score` tokenizer would delete all Hindi characters.
- **mBERTScore**: `bert_score` with multilingual BERT (`bert-base-multilingual-cased`, layer 9) → Precision, Recall, F1.

---

## Results (100 random Hindi test articles, seed 42, F1 × 100)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | mBERTScore-P | mBERTScore-R | mBERTScore-F1 | Avg. length (words) |
|-------|--------:|--------:|--------:|-------------:|-------------:|--------------:|--------------------:|
| **mT5** | **35.96** | **15.89** | **30.48** | **78.45** | **75.11** | **76.68** | 19.4 |
| mBART-50 | 35.20 | 13.71 | 29.17 | 77.87 | 74.97 | 76.34 | 20.0 |

Reference summaries average 27.3 words. Head-to-head, mT5 has the higher ROUGE-L on 52 articles and mBART on 47 (1 tie); mT5 has the higher mBERTScore on 54 and mBART on 46.

**Observations:** mT5 is slightly better on every metric (clearest on ROUGE-2), but the two models are close. Both produce fluent abstractive summaries and occasionally hallucinate details.

---

## How to Run

**Locally (uv):** `uv sync` in the repository root, then run the notebook with the `.venv` kernel. On CPU it evaluates 100 articles (~9 min per model, ~2.4 GB download per model on first run).

**Google Colab / Kaggle (faster):** upload the notebook, choose a **GPU runtime** and *Run all*. The setup cell installs `rouge-score`, `bert-score` and `sentencepiece` if they are missing. On a GPU the notebook automatically evaluates **1,000 articles** with batch size 16; set `N_EVAL = None` for the full test split.

**Optional fine-tuning:** set `RUN_FINETUNE = True` (GPU required) to fine-tune the base `google/mt5-small` and `facebook/mbart-large-50` models on Hindi XL-Sum yourself.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Abstractive summarization** | Generates a new, shorter text that paraphrases the source |
| **mT5** | Multilingual T5 – text-to-text Transformer pretrained with span corruption on mC4 (101 languages) |
| **mBART** | Multilingual denoising auto-encoder pretrained on 25/50 languages, uses language-ID tokens |
| **Beam search** | Keeps the *k* most probable partial sequences at each decoding step |
| **ROUGE-N / ROUGE-L** | n-gram overlap / longest-common-subsequence overlap with the reference |
| **BERTScore** | Token-level cosine similarity of contextual embeddings (here multilingual BERT) |

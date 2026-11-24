# Computational Linguistics and Natural Language Processing (CLNLP) - AIML

Lab Experiments for Computational Linguistics and Natural Language Processing.

## Directory Structure

- [Experiment-1/](Experiment-1) - Data Exploration and Visualization notebooks and datasets. For an in-depth breakdown of each notebook, see the **[Experiment 1 Detailed README](Experiment-1/README.md)**.
- [Experiment-2/](Experiment-2) - Natural Language Processing (NLP) Foundations notebooks and datasets. For an in-depth breakdown of each notebook, see the **[Experiment 2 Detailed README](Experiment-2/README.md)**.
- [Experiment-3/](Experiment-3) - Stemming, Lemmatization, and Regular Expressions. For an in-depth breakdown, see the **[Experiment 3 Detailed README](Experiment-3/README.md)**.
- [Experiment-4/](Experiment-4) - Term-Frequency Analysis, Named Entity Recognition, and TF-IDF. For an in-depth breakdown, see the **[Experiment 4 Detailed README](Experiment-4/README.md)**.
- [Experiment-5/](Experiment-5) - Subword Tokenization (BPE & SentencePiece), POS Tagging, and Word2Vec. For an in-depth breakdown, see the **[Experiment 5 Detailed README](Experiment-5/README.md)**.
- [Experiment-6/](Experiment-6) - GloVe Word Embeddings (from scratch & pretrained) and N-gram Language Model. For an in-depth breakdown, see the **[Experiment 6 Detailed README](Experiment-6/README.md)**.
- [Experiment-7/](Experiment-7) - Sentiment Analysis on IMDB with Logistic Regression, LSTM, Transformer and BERT. For an in-depth breakdown, see the **[Experiment 7 Detailed README](Experiment-7/README.md)**.
- [Experiment-8/](Experiment-8) - Abstractive Text Summarization with mT5 and mBART on Hindi XL-Sum. For an in-depth breakdown, see the **[Experiment 8 Detailed README](Experiment-8/README.md)**.
- [Experiment-9/](Experiment-9) - Authorship attribution of LLM-generated text (Kaggle *Name That LLM*): TF-IDF + stylometry, external data and LoRA-fine-tuned Qwen3 readers. For an in-depth breakdown, see the **[Experiment 9 Detailed README](Experiment-9/NameThatLLM/README.md)**.
- [Experiment-10/](Experiment-10) - Low-resource Neural Machine Translation on the WMT26 Indic MT task (English ↔ Assamese, Manipuri, Mizo, Bodo): Transformer baseline, back-translation and NLLB-200 fine-tuning. For an in-depth breakdown, see the **[Experiment 10 Detailed README](Experiment-10/Neural_Machine_Translation_%28NMT%29/README.md)**.
- [CLNLP_Lab_Report.docx](CLNLP_Lab_Report.docx) - Lab report (title page, contents, and aim / abstract / code description / code or repository link / results for Experiments 1–10).
- [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) - Python dependencies managed with `uv`.
- [LICENSE](LICENSE) - License documentation.
- [README.md](README.md) - This documentation.

---

## Experiments Details

### Experiment-1: Data Exploration and Visualization
*For a complete guide on how to run these, refer to the [Experiment-1 README](Experiment-1/README.md).*

Using the sample dataset [employee_information_100.csv](Experiment-1/employee_information_100.csv), these notebooks cover:
* **[1.1: Average Salary by Department](Experiment-1/experiment1.1.ipynb)**: Calculates the average salary per department and plots a bar chart.
* **[1.2: Employee Count by Department](Experiment-1/experiment1.2.ipynb)**: Displays the headcount for each department.
* **[1.3: Gender Ratio](Experiment-1/experiment1.3.ipynb)**: Visualizes the percentage of male and female employees using a pie chart.
* **[1.4: Salary Distribution](Experiment-1/experiment1.4.ipynb)**: Shows the distribution of salaries using a histogram.
* **[1.5: Experience vs. Salary](Experiment-1/experiment1.5.ipynb)**: Identifies patterns between work experience and compensation via a scatter plot.
* **[1.6: Top Earners](Experiment-1/experiment1.6.ipynb)**: Details the top 10 employees earning the highest salary.
* **[1.7: Maximum Salary by Department](Experiment-1/experiment1.7.ipynb)**: Discovers the maximum salary in each department.
* **[1.8: Above Average Salary](Experiment-1/experiment1.8.ipynb)**: Filters and displays employees who earn more than the overall average salary.
* **[1.9: Average Experience by Department](Experiment-1/experiment1.9.ipynb)**: Computes department-level average years of experience.
* **[1.10: Age Distribution](Experiment-1/experiment1.10.ipynb)**: Displays employee age distribution using a histogram.

### Experiment-2: Natural Language Processing (NLP) Foundations
*For a complete guide on how to run these, refer to the [Experiment-2 README](Experiment-2/README.md).*

Using clean text datasets, these notebooks cover:
* **[2.1: Basic Text Preprocessing](Experiment-2/2.1.ipynb)**: Converts text to lowercase, removes punctuation, numbers, and extra whitespace, displaying basic counts and cleaning statistics.
* **[2.2: Word & Sentence Tokenization](Experiment-2/2.2.ipynb)**: Implements sentence and word tokenizers using NLTK, spaCy, and a custom python implementation without external NLP libraries.
* **[2.3: Stop Word Removal](Experiment-2/2.3.ipynb)**: Tokenizes input text, filters out commonly occurring English stop words, and displays the original tokens, filtered tokens, and stop words separately.

### Experiment-3: Stemming, Lemmatization, and Regular Expressions
*For a complete guide on how to run these, refer to the [Experiment-3 README](Experiment-3/README.md).*

Using word lists and a sample text, these notebooks cover:
* **[3.1: Porter Stemmer](Experiment-3/lab3.ipynb)**: Applies the Porter Stemmer algorithm to a set of words, displaying each original word alongside its stemmed form.
* **[3.2: WordNet Lemmatization](Experiment-3/lab3.ipynb)**: Performs lemmatization using NLTK's `WordNetLemmatizer` with POS tags, converting inflected forms to their base dictionary form.
* **[3.3: Regex Information Extraction](Experiment-3/lab3.ipynb)**: Uses Regular Expressions to extract email addresses, URLs, mobile numbers, hashtags, and mentions from a sample text.

### Experiment-4: Term-Frequency Analysis, Named Entity Recognition & TF-IDF
*For a complete guide on how to run these, refer to the [Experiment-4 README](Experiment-4/README.md).*

Using a comprehensive NLP textbook-style passage, these experiments cover:
* **[4.1: TF Analysis & NER with Toolkit](Experiment-4/main.ipynb)**: Computes term frequencies using NLTK's `word_tokenize` and `Counter`, saves results to CSV, and performs Named Entity Recognition using spaCy's `en_core_web_sm` model.
* **[4.2: TF Analysis without Toolkit](Experiment-4/main.ipynb)**: Implements the same term-frequency analysis using only Python built-ins (`re`, `csv`, dictionary), with manual lowercasing, regex-based word extraction, and frequency sorting.
* **[4.3: TF-IDF from Scratch](Experiment-4/main.ipynb)**: Computes TF, DF, IDF, and TF-IDF scores for multiple documents without any NLP library, and identifies the top 10 terms by TF-IDF score per document.

### Experiment-5: Subword Tokenization, POS Tagging & Word2Vec
*For a complete guide on how to run these, refer to the [Experiment-5 README](Experiment-5/README.md).*

Using [input_sub_word_data.txt](Experiment-5/input_sub_word_data.txt) and a separate Word2Vec corpus (`corpus.txt`, not included), the notebook [Experiment_5.ipynb](Experiment-5/Experiment_5.ipynb) covers:
* **5.1: BPE Tokenization**: Pretrained GPT-2 tokenizer (`transformers`) vs. a from-scratch BPE with 50 merge steps, showing tokens and token IDs.
* **5.2: SentencePiece Tokenization**: Pretrained `mt5-small` model vs. a unigram model trained on the input text (`vocab_size=494`).
* **5.3: POS Tagging**: Tags a sample sentence using spaCy and NLTK, with tag descriptions.
* **5.4: POS Tagging with Frequency**: spaCy POS tags, descriptions, and token frequencies for the input text.
* **5.5: Word2Vec CBOW**: Trains CBOW with `gensim`, shows the vector for "learning" and its 3 most similar words.
* **5.6: Word2Vec Skip-Gram**: Trains Skip-Gram, shows the vector for "language" and its 5 most similar words.
* **5.7: Cosine Similarity**: Computes similarity between word pairs (machine–learning, natural–language, deep–learning).

### Experiment-6: GloVe Word Embeddings & N-gram Language Model
*For a complete guide on how to run these, refer to the [Experiment-6 README](Experiment-6/README.md).*

The notebook [Experiment_6.ipynb](Experiment-6/Experiment_6.ipynb) covers:
* **6.1: GloVe from Scratch**: Builds a distance-weighted co-occurrence matrix from [glove_corpus.txt](Experiment-6/glove_corpus.txt) and trains GloVe (weighted least squares + AdaGrad) in NumPy; finds the words most similar to "king" (queen, son, wise, …).
* **6.2: Pretrained GloVe (Gensim)**: Loads `glove-wiki-gigaword-100`, lists the neighbours of "king" (prince, queen, monarch, throne, …) and solves *king − man + woman ≈ queen*.
* **6.3: N-gram Language Model**: Unigram/bigram/trigram model from scratch on [ngram_corpus.txt](Experiment-6/ngram_corpus.txt) with MLE and Laplace smoothing, next-word prediction with back-off, perplexity and text generation.

### Experiment-7: Sentiment Analysis – Comparative Study
*For a complete guide on how to run these, refer to the [Experiment-7 README](Experiment-7/README.md).*

The notebook [Experiment_7.ipynb](Experiment-7/Experiment_7.ipynb) trains and compares four models on the IMDB 50K Movie Reviews dataset:
* **Logistic Regression** on TF-IDF uni+bi-grams.
* **Bidirectional LSTM** trained from scratch (PyTorch).
* **Vanilla Transformer encoder** trained from scratch (PyTorch).
* **BERT** (`bert-base-uncased`) fine-tuned for classification.

Evaluation uses Accuracy, Precision, Recall, F1-score and Confusion Matrices on the full 10,000-review test set (run on a Google Colab L4 GPU):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9020 | 0.8954 | 0.9104 | 0.9028 |
| LSTM | 0.8504 | 0.8194 | 0.8990 | 0.8573 |
| Transformer | 0.8386 | 0.8103 | 0.8842 | 0.8456 |
| **BERT** | **0.9252** | **0.9254** | **0.9250** | **0.9252** |

### Experiment-8: Abstractive Summarization with mT5 & mBART (Hindi XL-Sum)
*For a complete guide on how to run these, refer to the [Experiment-8 README](Experiment-8/README.md).*

The notebook [Experiment_8.ipynb](Experiment-8/Experiment_8.ipynb) summarises Hindi BBC articles from the XL-Sum dataset with `csebuetnlp/mT5_multilingual_XLSum` (mT5) and `mskayacioglu/mbart-xlsum-2` (mBART-50) and compares them with ROUGE-1/2/L (Devanagari-aware tokenizer) and mBERT-based BERTScore.

| Model (100 Hindi test articles) | ROUGE-1 | ROUGE-2 | ROUGE-L | mBERTScore-F1 |
|---|---|---|---|---|
| **mT5** | **35.96** | **15.89** | **30.48** | **76.68** |
| mBART-50 | 35.20 | 13.71 | 29.17 | 76.34 |

### Experiment-9: Authorship Attribution of LLM Text (Kaggle: Name That LLM)
*For the full method, results and file guide, refer to the [Experiment-9 README](Experiment-9/NameThatLLM/README.md).* Team: Somansh Goel, Amulya Jain.

Given a 40–250-word passage, predict which of **18 LLMs** wrote it (balanced accuracy, chance 0.056). Test prompts are unseen, so models are evaluated with grouped 5-fold CV by prompt:
* **[ntl_baseline.py](Experiment-9/NameThatLLM/ntl_baseline.py)**: four TF-IDF views (char, word, punctuation shape, function words) + 35 stylometric features, log-linear blend of logistic regressions; optionally trained with the 231k-passage public Open Pool.
* **[hpc/train_lora.py](Experiment-9/NameThatLLM/hpc/train_lora.py)**: truncated Qwen3-8B/14B-Base with LoRA and a multi-layer mean-pooling head, trained on the UPES HPC (H100); **[kaggle_lora/infer_lora.py](Experiment-9/NameThatLLM/kaggle_lora/infer_lora.py)** runs it on Kaggle T4s.

| Model (grouped CV) | Balanced accuracy |
|---|---|
| TF-IDF + stylometry, train only | 0.389 |
| TF-IDF + Open Pool + stylometry | 0.514 |
| Qwen3-8B LoRA reader (fold 0) | 0.623 |
| **8B + 14B readers + TF-IDF (fold 0)** | **0.653** |

### Experiment-10: Low-Resource Neural Machine Translation (WMT26 Indic MT)
*For setup, commands, results and figures, refer to the [Experiment-10 README](Experiment-10/Neural_Machine_Translation_%28NMT%29/README.md).* Team: Amulya Jain, Himanshu Pokhriyal, Anirvan Ranjan, Narind Verma, Somansh Goel.

English ↔ Assamese, Manipuri, Mizo and Bodo (8 directions) from the WMT26 Indic MT 2026 release, as a five-stage pipeline (data → preprocessing/EDA → baseline → advanced models → evaluation):
* **Transformer baseline** trained from scratch (PyTorch, 9.4M parameters, joint SentencePiece per pair).
* **+ Back-translation** with tagged synthetic data generated by the baseline.
* **NLLB-200-distilled-600M** fine-tuned (Bodo through the Hindi `hin_Deva` proxy tag).

Evaluation: chrF++ (primary), BLEU and COMET with 1,000-sample bootstrap CIs and paired significance tests; GPU training on Colab/Kaggle notebooks. It also includes a [presentation website](Experiment-10/Neural_Machine_Translation_%28NMT%29/presentation/index.html) and [slides](Experiment-10/Neural_Machine_Translation_%28NMT%29/presentation/NMT_Presentation.pptx).

| Average over 8 directions (test) | chrF++ | BLEU | COMET |
|---|---|---|---|
| Transformer baseline | 39.91 | 17.24 | 64.59 |
| + Back-translation | 39.76 | 17.17 | 64.10 |
| **NLLB-200 fine-tuned** | **52.37** | **28.48** | **74.33** |

---

## Requirements

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`, Python 3.12). Main packages: pandas, numpy, matplotlib, seaborn, jupyter, nltk, spacy, scikit-learn, gensim, torch, transformers, datasets, sentencepiece, rouge-score, bert-score.

Create the virtual environment (`.venv`) and install everything with:
```bash
uv sync
uv run python -m spacy download en_core_web_sm   # spaCy model used in Experiments 2, 4 and 5
```

Experiment 10 has its own pinned environment: `cd "Experiment-10/Neural_Machine_Translation_(NMT)" && bash run.sh setup` creates a local, gitignored `.venv` with uv (see its README). Experiment 9's LoRA scripts need a GPU server or Kaggle (see its README and NOTES.md).

Alternatively, with pip:
```bash
pip install pandas numpy matplotlib seaborn jupyter nltk spacy scikit-learn gensim torch transformers datasets sentencepiece rouge-score bert-score
python -m spacy download en_core_web_sm
```

---

## How to Run

1. Open the desired notebook in Jupyter Notebook, JupyterLab, or VS Code (select the `.venv` kernel).
2. Run the cells in order.
3. Make sure the input dataset or text file is in the same directory as the notebook you are running.
4. **Google Colab / Kaggle:** Experiments 6–8 include a setup cell that installs missing packages and fetches input files automatically. Experiments 7 and 8 are compute-heavy – choose a **GPU runtime**; the notebooks detect the GPU and switch from the reduced CPU settings (BERT training subset, number of summarised articles) to the full dataset.

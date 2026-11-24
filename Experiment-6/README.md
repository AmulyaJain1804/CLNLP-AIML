# Experiment 6: GloVe Word Embeddings & N-gram Language Model

This folder contains a Jupyter Notebook that implements **GloVe word embeddings** (from scratch and pretrained via Gensim) and a **statistical N-gram language model** from scratch.

## Files

| File | Description |
|------|-------------|
| [`Experiment_6.ipynb`](Experiment_6.ipynb) | Notebook containing all sub-experiments (6.1 – 6.3), executed with outputs |
| [`glove_corpus.txt`](glove_corpus.txt) | 24-sentence corpus used to train GloVe from scratch (6.1) |
| [`ngram_corpus.txt`](ngram_corpus.txt) | Corpus used for the N-gram language model (6.3) |
| [`Experiment-6.docx`](Experiment-6.docx) | Problem statement |

---

## Lab Experiments

### Experiment 6.1: GloVe from Scratch (no pretrained model)
**Input:** `glove_corpus.txt`

1. Lower-case, tokenise (`[a-z]+`) and remove a small stop-word set → vocabulary of 61 words.
2. Build a symmetric **co-occurrence matrix** with a window of 4, weighting each pair by `1/distance`.
3. Train GloVe with the weighted least-squares loss
   `J = Σ f(Xij) (wi·w̃j + bi + b̃j − log Xij)²`, `f(x) = min((x/x_max)^0.75, 1)` with `x_max = 10`, optimised with **AdaGrad** (50-d vectors, 500 epochs).
4. Final embedding = `W + W̃`; rank neighbours by **cosine similarity**; PCA plot of the embeddings.

**Result – most similar to "king":** queen (0.996), son (0.964), wise (0.940), powerful (0.935), lives (0.901).

### Experiment 6.2: GloVe using a Pretrained Model (Gensim)
Loads `glove-wiki-gigaword-100` (400K words × 100-d, Wikipedia 2014 + Gigaword 5) with `gensim.downloader` (~128 MB on first run).

**Result – most similar to "king":** prince (0.768), queen (0.751), son (0.702), brother (0.699), monarch (0.698), throne, kingdom, father, emperor, ii.
Analogy `king − man + woman` → **queen** (0.770).

### Experiment 6.3: N-gram Language Model from Scratch
**Input:** `ngram_corpus.txt` (8 sentences)

- Sentences padded with `<s>` / `</s>`; unigram, bigram and trigram counts.
- **MLE** and **Laplace-smoothed** conditional probabilities.
- **Next-word prediction** with back-off (trigram → bigram → unigram), e.g. `natural language → processing (1.00)`, `artificial intelligence → </s> (0.67), includes (0.33)`.
- Sentence probability and **perplexity** (corpus sentence ≈ 11.7 vs. scrambled sentence ≈ 37.9 with bigrams).
- Greedy and sampled **text generation**.

**Libraries:** `numpy`, `pandas`, `matplotlib`, `gensim`, `re`, `collections`

---

## How to Run

**Locally (uv):** from the repository root run `uv sync`, then open the notebook with the `.venv` kernel and run all cells.

**Google Colab / Kaggle:** open the notebook and run all cells – the first setup cell installs `gensim` if missing and downloads the two corpus files from this repository (or upload them manually).

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Co-occurrence matrix** | Counts how often word pairs appear within a context window |
| **GloVe** | Learns vectors whose dot product approximates log co-occurrence counts |
| **Cosine similarity** | Angle-based similarity between two word vectors |
| **N-gram LM** | Estimates P(word \| previous n−1 words) from counts |
| **Laplace smoothing** | Adds 1 to every count to avoid zero probabilities |
| **Back-off** | Falls back to a lower-order model when a context is unseen |
| **Perplexity** | Inverse probability of a sentence normalised by its length (lower = better) |

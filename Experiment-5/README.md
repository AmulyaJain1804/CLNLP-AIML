# Experiment 5: Subword Tokenization, POS Tagging & Word2Vec

This folder contains a Jupyter Notebook (written for Google Colab) covering **subword tokenization (BPE & SentencePiece)**, **Part-of-Speech (POS) tagging**, and **Word2Vec word embeddings (CBOW & Skip-Gram)**.

## Files

| File | Description |
|------|-------------|
| [`Experiment_5.ipynb`](Experiment_5.ipynb) | Notebook containing all sub-experiments (5.1 – 5.7) |
| [`input_sub_word_data.txt`](input_sub_word_data.txt) | Input text used for tokenization (5.1, 5.2) and POS frequency (5.4) |

> **Note:** Experiments 5.5 – 5.7 read a separate corpus file (`corpus.txt`, one sentence per line) that is **not included** in this folder. Provide your own and update the path in the notebook.

---

## Lab Experiments

### Experiment 5.1: BPE Subword Tokenization
**Input:** `input_sub_word_data.txt`

- **(a) Pretrained:** Uses Hugging Face `GPT2Tokenizer` (`gpt2`) to produce BPE tokens and token IDs.
- **(b) From scratch:** Implements BPE manually:
  1. Extract lowercase words with `re.findall(r"[A-Za-z]+")`.
  2. Split each word into characters plus an end-of-word marker `</w>`; count word frequencies.
  3. Repeat **50 merge steps**: count adjacent symbol pairs, merge the most frequent pair everywhere.
  4. Build the subword vocabulary, map tokens to IDs, and print tokens with IDs for the first 50 words.

**Libraries:** `transformers`, `re`, `collections.Counter`

### Experiment 5.2: SentencePiece Tokenization
- **(a) Pretrained:** Loads the `google/mt5-small` `spiece.model` and prints each token with its ID.
- **(b) Without pretrained model:** Trains a **unigram** SentencePiece model (`vocab_size=494`) on the input text, then encodes the text and prints tokens with IDs.

**Libraries:** `sentencepiece`

### Experiment 5.3: POS Tagging on a Sentence
Sentence: *"The young student is reading an interesting book in the library."*
- **(a) spaCy** (`en_core_web_sm`): prints unique tokens, universal POS tag and description via `spacy.explain`.
- **(b) NLTK:** `word_tokenize` + `pos_tag` (Penn Treebank tags), with descriptions via `nltk.help.upenn_tagset`.

### Experiment 5.4: POS Tagging with Frequency (spaCy)
Runs spaCy over the input file and prints each unique token (lowercased, punctuation and whitespace excluded) with its POS tag, tag description, and frequency.

### Experiment 5.5: Word2Vec – CBOW
Trains `gensim` Word2Vec with `sg=0` (CBOW; `vector_size=100`, `window=5`, `min_count=1`). Displays the vector for **"learning"** and its **3 most similar words**.

### Experiment 5.6: Word2Vec – Skip-Gram
Trains Word2Vec with `sg=1` on the same corpus. Displays the vector for **"language"** and its **5 most similar words**.

### Experiment 5.7: Cosine Similarity of Word Pairs
Trains a CBOW model and computes cosine similarity for the pairs (`machine`, `learning`), (`natural`, `language`), and (`deep`, `learning`).

---

## How to Run

1. Install dependencies:
   ```bash
   pip install transformers sentencepiece spacy nltk gensim
   python -m spacy download en_core_web_sm
   ```
2. Open `Experiment_5.ipynb` in Google Colab (or Jupyter / VS Code).
3. Upload `input_sub_word_data.txt` (and your `corpus.txt` for 5.5 – 5.7) and adjust the file paths (the notebook uses `/content/...`).
4. Note that 5.1(b) uses `re` and `Counter`; make sure `import re` and `from collections import Counter` are executed first.
5. Run cells sequentially.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **BPE** | Iteratively merges the most frequent adjacent symbol pair to build a subword vocabulary |
| **SentencePiece** | Language-independent subword tokenizer (here: unigram model) |
| **POS Tagging** | Assigning grammatical categories (noun, verb, ...) to tokens |
| **CBOW** | Word2Vec variant predicting a word from its context |
| **Skip-Gram** | Word2Vec variant predicting context words from a word |
| **Cosine Similarity** | Measures the angle between two word vectors as semantic similarity |

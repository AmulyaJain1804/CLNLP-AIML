# Experiment 3: Stemming, Lemmatization and Regular Expressions

This folder contains a Jupyter Notebook covering two **text normalisation** techniques, **stemming** (Porter Stemmer) and **lemmatization** (WordNet), and **information extraction with Regular Expressions** (emails, URLs, mobile numbers, hashtags and mentions).

## Files

| File | Description |
|------|-------------|
| [`lab3.ipynb`](lab3.ipynb) | Notebook containing all three sub-experiments (3.1, 3.2, 3.3) |

All inputs (word lists and the sample text) are defined inside the notebook, so no extra data files are needed.

---

## Lab Experiments

### Experiment 3.1: Stemming with the Porter Stemmer

**Input:** playing, played, plays, studies, studying, connected, connection, computers (plus *Unhappiness*)

**Description:** Applies NLTK's **`PorterStemmer`**, which strips common English suffixes with fixed rules to reduce words to a root form.

The program:
1. Creates a `PorterStemmer` object.
2. Calls `stemmer.stem(word)` on each word in the list.
3. Prints every original word next to its stem.

**Output:**

| Original word | Stem |
|------|------|
| playing | play |
| played | play |
| plays | play |
| studies | studi |
| studying | studi |
| connected | connect |
| connection | connect |
| computers | comput |
| Unhappiness | unhappi |

> **Note:** Stems need not be real words (*studi*, *comput*, *unhappi*). The stemmer only cuts suffixes, so related forms map to the same stem, which is what search and indexing need.

**Libraries used:** `nltk`

---

### Experiment 3.2: Lemmatization with WordNet

**Input:** cats, dogs, running, runs, ran, studies, studying, better, children, mice, went, ate, leaves, caring (plus *unhappiness*)

**Description:** Uses NLTK's **`WordNetLemmatizer`** with a part-of-speech tag for every word (`n` noun, `v` verb, `a` adjective), so each word is mapped to its dictionary form (lemma).

The program:
1. Downloads the `wordnet` and `omw-1.4` corpora.
2. Defines the word list and a dictionary of POS tags.
3. Calls `lemmatizer.lemmatize(word, pos=...)` for each word.
4. Prints every original word next to its lemma.

**Output:**

| Original word | POS | Lemma |
|------|------|------|
| cats | n | cat |
| dogs | n | dog |
| running | v | run |
| runs | v | run |
| ran | v | run |
| studies | v | study |
| studying | v | study |
| better | a | good |
| children | n | child |
| mice | n | mouse |
| went | v | go |
| ate | v | eat |
| leaves | n | leaf |
| caring | v | care |
| unhappiness | n | unhappiness |

> **Note:** Unlike stemming, lemmatization returns real words and handles irregular forms (*ran → run*, *went → go*, *mice → mouse*, *better → good*). It needs the correct POS tag: without `pos="v"`, *running* would stay unchanged.

**Libraries used:** `nltk` (WordNet)

---

### Experiment 3.3: Information Extraction with Regular Expressions

**Input:** a sample workshop announcement (defined in the notebook) containing URLs, email addresses, phone numbers, hashtags and social-media mentions.

**Description:** Uses Python's **`re.findall()`** with one pattern per item type:

| Item | Pattern |
|------|---------|
| Email addresses | `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` |
| URLs | `(?:https?://\|www\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/…)?` |
| Mobile numbers | `(?:\+91[-\s]?)?[6-9]\d{9}` (optional +91 code, 10 digits starting 6–9) |
| Hashtags | `#[A-Za-z0-9_]+` |
| Mentions | `@[A-Za-z0-9_]+` |

**Output:**

| Category | Extracted |
|------|------|
| Emails | nlpworkshop@gmail.com, support@python.org |
| URLs | https://www.nlpworkshop.com, www.python.org, https://github.com/NLPWorkshop |
| Mobile numbers | +91-9876543210, 9123456789 |
| Hashtags | #NLP, #Python, #MachineLearning |
| Mentions | @NLPWorkshop, @PythonLearner (plus `@gmail` and `@python`, see note) |

> **Note:** The simple mention pattern also matches the domain part of the email addresses (`@gmail`, `@python`). A lookbehind such as `(?<![\w.])@[A-Za-z0-9_]+` limits it to real mentions.

**Libraries used:** `re` (Python built-in)

---

## How to Run

1. Install the required library:
   ```bash
   pip install nltk
   ```
   (or `uv sync` from the repository root)
2. Navigate to the `Experiment-3/` folder.
3. Open `lab3.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells from top to bottom. Experiment 3.2 downloads the WordNet data on first run.

---

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Stemming** | Rule-based suffix stripping to a common root; fast, but the stem may not be a real word |
| **Lemmatization** | Dictionary-based reduction to the base form (lemma) using vocabulary and part of speech |
| **Porter Stemmer** | Classic English stemming algorithm (Porter, 1980) applying suffix rules in steps |
| **WordNet** | Lexical database of English used by NLTK to look up lemmas |
| **Regular Expressions** | Patterns that describe text, used to search, extract and validate strings |

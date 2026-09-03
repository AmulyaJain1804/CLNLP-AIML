# Computational Linguistics and Natural Language Processing (CLNLP) - AIML

Lab Experiments for Computational Linguistics and Natural Language Processing.

## Directory Structure

- [Experiment-1/](Experiment-1) - Data Exploration and Visualization notebooks and datasets. For an in-depth breakdown of each notebook, see the **[Experiment 1 Detailed README](Experiment-1/README.md)**.
- [Experiment-2/](Experiment-2) - Natural Language Processing (NLP) Foundations notebooks and datasets. For an in-depth breakdown of each notebook, see the **[Experiment 2 Detailed README](Experiment-2/README.md)**.
- [Experiment-3/](Experiment-3) - Stemming, Lemmatization, and Regular Expressions. For an in-depth breakdown, see the **[Experiment 3 Detailed README](Experiment-3/README.md)**.
- [Experiment-4/](Experiment-4) - Term-Frequency Analysis, Named Entity Recognition, and TF-IDF. For an in-depth breakdown, see the **[Experiment 4 Detailed README](Experiment-4/README.md)**.
- [Experiment-5/](Experiment-5) - Subword Tokenization using Byte Pair Encoding (BPE). For an in-depth breakdown, see the **[Experiment 5 Detailed README](Experiment-5/README.md)**.
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

### Experiment-5: Subword Tokenization using Byte Pair Encoding (BPE)
*For a complete guide on how to run these, refer to the [Experiment-5 README](Experiment-5/README.md).*

Using an NLP corpus on tokenization and language models, this experiment covers:
* **[5: BPE Subword Tokenization from Scratch](Experiment-5/main.ipynb)**: Implements the Byte Pair Encoding algorithm without any pre-trained tokenizer — learning subword merge rules from the corpus, building a vocabulary, tokenizing unseen words, assigning token IDs, and evaluating tokenizer statistics across different numbers of merge steps.

---

## Requirements

Ensure you have the following packages installed:
* Python 3
* pandas
* matplotlib
* jupyter
* nltk
* spacy

To install the dependencies, you can run:
```bash
pip install pandas matplotlib jupyter nltk spacy
python -m spacy download en_core_web_sm
```

---

## How to Run

1. Open the desired notebook in Jupyter Notebook, JupyterLab, or VS Code.
2. Run the cells in order.
3. Make sure the input dataset or text file is in the same directory as the notebook you are running.

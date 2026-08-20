# Experiment 2: Natural Language Processing (NLP) Foundations

This folder contains Jupyter Notebooks focused on the foundational concepts of Natural Language Processing (NLP). The experiments involve taking unstructured text data and applying various preprocessing steps to prepare it for analysis or machine learning models.

## Lab Experiments

* **[Experiment 2.1: Basic Text Preprocessing](2.1.ipynb)**
  **Input:** [2.1_text_data.txt](2.1_text_data.txt)
  **Description:** This notebook performs basic text preprocessing on a given raw text data. The program:
  1. Converts the entire text to lowercase (and displays the count of uppercase characters).
  2. Removes all punctuation marks (and counts different types of punctuations).
  3. Removes all numerical digits.
  4. Cleans up extra whitespace.
  5. Displays the fully cleaned text ready for further NLP tasks.

* **[Experiment 2.2: Word & Sentence Tokenization](2.2.ipynb)**
  **Input:** [2.2_tokenization_data.txt](2.2_tokenization_data.txt)
  **Description:** This notebook explores tokenization—the process of breaking down text into smaller units (tokens). It performs word tokenization and sentence tokenization, displaying the resulting words and sentences separately. To provide a comprehensive understanding, the notebook implements the tokenization in three different ways:
  1. Using the classic `NLTK` library.
  2. Using the modern `spaCy` library.
  3. Using a custom Python implementation without relying on external NLP libraries.

* **[Experiment 2.3: Stop Word Removal](2.3.ipynb)**
  **Input:** [2.3_clean_data.txt](2.3_clean_data.txt)
  **Description:** This notebook focuses on removing stop words (commonly occurring words like "is", "the", "a", "an", "in", "of", "and", "to") from a tokenized text. Removing stop words reduces the dataset size and helps models focus on the most meaningful words. The notebook displays:
  1. The original tokens.
  2. The filtered tokens (without stop words).
  3. The stop words that were separated from the text.

---

## How to Run
1. Make sure you have installed the necessary NLP libraries: `pip install nltk spacy`.
2. Navigate to this folder.
3. Start Jupyter Notebook, JupyterLab, or VS Code.
4. Open the desired notebook (`2.1.ipynb`, `2.2.ipynb`, or `2.3.ipynb`).
5. Ensure the corresponding `.txt` file is in the same folder, and run the cells sequentially.

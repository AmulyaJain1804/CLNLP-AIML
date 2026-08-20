# CLNLP-AIML

Lab Experiments for Computational Linguistics and Natural Language Processing.

## Directory Structure

- [Experiment-1/](Experiment-1)
  - [employee_information_100.csv](Experiment-1/employee_information_100.csv) - Employee dataset used for Experiment-1.
  - [experiment1.1.ipynb](Experiment-1/experiment1.1.ipynb) - Average salary of each department (Bar Chart).
  - [experiment1.2.ipynb](Experiment-1/experiment1.2.ipynb) - Employee count per department.
  - [experiment1.3.ipynb](Experiment-1/experiment1.3.ipynb) - Gender ratio visualization (Pie Chart).
  - [experiment1.4.ipynb](Experiment-1/experiment1.4.ipynb) - Salary distribution (Histogram).
  - [experiment1.5.ipynb](Experiment-1/experiment1.5.ipynb) - Experience vs Salary relationship (Scatter Plot).
  - [experiment1.6.ipynb](Experiment-1/experiment1.6.ipynb) - Top 10 highest-paid employees.
  - [experiment1.7.ipynb](Experiment-1/experiment1.7.ipynb) - Highest salary per department.
  - [experiment1.8.ipynb](Experiment-1/experiment1.8.ipynb) - Employees earning more than the overall average.
  - [experiment1.9.ipynb](Experiment-1/experiment1.9.ipynb) - Average experience per department.
  - [experiment1.10.ipynb](Experiment-1/experiment1.10.ipynb) - Age distribution (Histogram).
- [Experiment-2/](Experiment-2)
  - [2.1_text_data.txt](Experiment-2/2.1_text_data.txt) - Sample text file for preprocessing.
  - [2.1.ipynb](Experiment-2/2.1.ipynb) - Text preprocessing (lowercasing, punctuation, number, and whitespace removal).
  - [2.2_tokenization_data.txt](Experiment-2/2.2_tokenization_data.txt) - Sample text file for tokenization.
  - [2.2.ipynb](Experiment-2/2.2.ipynb) - Word and Sentence Tokenization (NLTK, spaCy, and custom implementation).
  - [2.3_clean_data.txt](Experiment-2/2.3_clean_data.txt) - Sample text file for stop words removal.
  - [2.3.ipynb](Experiment-2/2.3.ipynb) - Tokenization and Stop Word Removal.
- [LICENSE](LICENSE) - License documentation.
- [README.md](README.md) - This documentation.

---

## Experiments Details

### Experiment-1: Data Exploration and Visualization

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

Using clean text datasets, these notebooks cover:
* **[2.1: Basic Text Preprocessing](Experiment-2/2.1.ipynb)**: Converts text to lowercase, removes punctuation, numbers, and extra whitespace, displaying basic counts and cleaning statistics.
* **[2.2: Word & Sentence Tokenization](Experiment-2/2.2.ipynb)**: Implements sentence and word tokenizers using NLTK, spaCy, and a custom python implementation without external NLP libraries.
* **[2.3: Stop Word Removal](Experiment-2/2.3.ipynb)**: Tokenizes input text, filters out commonly occurring English stop words, and displays the original tokens, filtered tokens, and stop words separately.

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
```

---

## How to Run

1. Open the desired notebook in Jupyter Notebook, JupyterLab, or VS Code.
2. Run the cells in order.
3. Make sure the input dataset or text file is in the same directory as the notebook you are running.

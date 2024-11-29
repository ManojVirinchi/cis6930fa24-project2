# Unredactor

## Project Overview

This project focuses on creating a **name unredaction system** using a **Multinomial Naive Bayes (MultinomialNB)** classifier. The goal is to train a machine learning model that can identify and predict redacted names in a given context by analyzing text data. The system leverages both **textual features** (e.g., n-grams, character-level analysis) and **custom features** (e.g., sentence counts, redaction patterns) to make predictions.

## Installation

To set up the environment and run the project, follow these steps:

1. **Install dependencies:**

   Ensure Python 3.12, create the pipfile and pipfile.lock, then set up a virtual environment:
   ```bash
   pipenv install
   pipenv shell
   ```

   The following are the necessary packages:

   - `pandas`, `numpy` - Data handling and numerical operations
   - `scikit-learn` - Machine learning tools
   - `spaCy`, `nltk` - Natural language processing tasks
   - `argparse` - Command-line argument parsing
   - `scipy` - Scientific computations
   - `re` - Regular expressions for pattern matching

2. Install the `en_core_web_lg` model for spaCy by running:
   ```bash
   python -m spacy download en_core_web_lg
   ```

3. **Download NLTK resources:**

   The script requires some NLTK resources for text preprocessing. These can be downloaded by running:
   ```python
   import nltk
   nltk.download('punkt_tab')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Data Preparation

Before training the model, a dataset of redacted names and contexts is required. To generate this dataset, we use the **`extract_names.py`** script, which extracts names from the **IMDB dataset** and redacts them. The extracted names are saved to a CSV file (`imdb_names.csv`), which is then combined with the main training dataset (`unredactor.tsv`) to create the final dataset used for training the model.

### **`extract_names.py` Script**

This script processes the IMDB reviews dataset to extract names of people and replace them with redacted symbols (█). It uses the `spaCy` library to detect named entities labeled as **PERSON** in the text. For each review, the name entities are replaced with a redacted version, and the context, along with the sentiment and other metadata, is saved.

The process works as follows:
- The script iterates through the IMDB directory structure (containing training and test data for positive and negative sentiments).
- For each text file, the script reads the content, processes it using `spaCy` to detect **PERSON** entities, and replaces the names with the redaction symbol.
- The relevant metadata, including the redacted context, original name, sentiment, and file path, are saved.
- Finally, the unique names are extracted and saved to the `imdb_names.csv` file, which is later used for training the unredaction model.

## Features of the Model

The model pipeline integrates several key steps, including:

1. **Text Feature Extraction:**
   - **CountVectorizer (Word & Character-based n-grams):** Extracts word and character-level n-grams (1 to 2 words or 2 to 4 characters) from the text.
   - **TF-IDF Vectorizer:** Transforms text data into a **Term Frequency-Inverse Document Frequency** (TF-IDF) matrix, highlighting the importance of words across the dataset.
   - These methods capture the frequency and presence of relevant terms and character sequences in the context.

2. **Custom Feature Extraction:**
   - The **`DictFeatureTransformer`** is used to extract various statistical and linguistic features from the text:
     - **Token Count:** The number of words in the context.
     - **Entity Count:** The number of named entities identified by **spaCy**.
     - **Redaction Length:** The total length of redacted sections in the text, identified by the █ symbol.
     - **Sentence Count:** The number of sentences in the context.
     - **Unique Words:** The number of unique words used in the context.
     - **Lexical Diversity:** A measure of the diversity of vocabulary in the text.
     - **Uppercase Words:** The count of uppercase words, which may indicate proper names or redactions.
     - **Average Word Length:** An average length of words in the context.
     - **Special Characters:** The number of special characters present in the text.

3. **Pipeline Construction:**
   - **FeatureUnion:** Combines various feature extraction methods into a single pipeline.
   - **MaxAbsScaler:** Scales features to the range [0, 1], preserving sparsity.
   - **AbsoluteTransformer:** Converts all feature values to their absolute values.
   - **Multinomial Naive Bayes (MultinomialNB):** The classifier used to predict the redacted names.

4. **Model Training and Evaluation:**
   - The model is trained using data from the provided training file and then evaluated using precision, recall, and F1-score.
   - The `train_model` function trains the model, while the `evaluate_model` function assesses its performance on a validation set.

5. **Unredaction:**
   - The `unredact` function uses the trained model to predict the original names for redacted contexts.

## Model Evaluation

The evaluation results show precision, recall, and F1-score for the model. These metrics are calculated using a **weighted average**, which takes into account the frequency of each class.

### Key Observations:
- The accuracy, precision, and recall values may be lower than expected. This is primarily due to the fact that many names in the dataset are quite rare, making it difficult for the model to learn strong patterns for each individual name. As a result, the model struggles to achieve high confidence in its predictions for these names.

## How the Code Works

### Main Components:
- **`preprocess` function:** Cleans and preprocesses the input text, including tokenization, stopword removal, and lemmatization.
- **`extract_features` function:** Extracts linguistic and statistical features from the text that are used for training and prediction.
- **`DictFeatureTransformer` class:** Custom transformer to process and extract features into a format suitable for input into a machine learning model.
- **`create_pipeline` function:** Constructs the full machine learning pipeline that includes feature extraction, scaling, and classification.
- **`train_model` function:** Trains the MultinomialNB classifier using the processed training data.
- **`evaluate_model` function:** Evaluates the model's performance on the validation data using precision, recall, and F1-score.
- **`unredact` function:** Given a context with redacted names, the function predicts the original names using the trained model.

### Running the Script:

To run the script, use the following command line interface:

```bash
python unredactor.py <train_file> <test_file> <submission_file>
```

Where:
- `train_file` is the path to the training dataset file (e.g., `unredactor.tsv`).
- `test_file` is the path to the test dataset file (e.g., `test.tsv`).

### Test Case Files
1. **`test_extract_features_basic`**: Validates feature extraction for a simple text with one redaction.  
2. **`test_extract_features_empty`**: Ensures all features return zero for an empty input string.  
3. **`test_extract_features_special_characters`**: Confirms the correct handling of special characters and redactions in text.  
4. **`test_create_pipeline`**: Verifies the structure and components of the machine learning pipeline.  


## Known Bugs and Limitations

- **Accuracy Issues:** The model's performance, particularly in terms of precision and recall, may not be optimal due to the rarity of names in the dataset. As a result, the classifier may struggle to develop a strong confidence level for each individual name, leading to low precision and recall scores.
  
## Key Assumptions

- **Named Entity Recognition Accuracy:**  
   The `spaCy` library accurately identifies and labels person names as named entities during preprocessing without significant false positives or negatives.

- **Feature Relevance:**  
   The extracted features (e.g., n-grams, token counts, lexical diversity) are sufficient to differentiate between redacted names and other possible redactions or placeholders.

- **Static Dataset Format:**  
   The structure and format of the training and testing datasets remain consistent, ensuring that the model can process and analyze the data without additional preprocessing modifications.



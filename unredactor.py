import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import spacy
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import os
import argparse
import re
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Static path configurations
IMDB_NAMES_FILE = "imdb_names.csv"

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# NLTK setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    if pd.isna(text):
        return ""
    
    # Tokenization
    tokens = word_tokenize(str(text))
    
    # Lowercasing, stopword removal, and lemmatization
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words and token.isalpha()]
    
    return ' '.join(tokens)

def extract_features(text):
    if pd.isna(text):
        return {feature: 0 for feature in ['num_words', 'num_entities', 'redaction_length', 'num_sentences', 'avg_word_length', 'num_unique_words', 'num_uppercase_words', 'lexical_diversity', 'num_redactions', 'avg_redaction_length', 'num_special_chars']}
    
    doc = nlp(str(text))
    
    words = [token for token in doc if token.is_alpha]
    num_words = len(words)
    num_sentences = len(list(doc.sents))
    num_entities = len(doc.ents)
    
    redactions = re.findall(r'█+', text)
    redaction_length = sum(len(r) for r in redactions)
    num_redactions = len(redactions)
    
    word_lengths = [len(token.text) for token in words]
    unique_words = set(token.text.lower() for token in words)
    
    features = {
        'num_words': num_words,
        'num_entities': num_entities,
        'redaction_length': redaction_length,
        'num_sentences': num_sentences,
        'avg_word_length': sum(word_lengths) / num_words if num_words else 0,
        'num_unique_words': len(unique_words),
        'num_uppercase_words': sum(1 for token in words if token.is_upper),
        'lexical_diversity': len(unique_words) / num_words if num_words else 0,
        'num_redactions': num_redactions,
        'avg_redaction_length': redaction_length / num_redactions if num_redactions else 0,
        'num_special_chars': len(re.findall(r'[^a-zA-Z0-9\s]', text))
    }
    
    return features

class DictFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, csr_matrix):
            X = X.toarray()
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = X[:, 0]
        features = [extract_features(str(text)) for text in X]
        return features

class AbsoluteTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.abs(X)

def create_pipeline():
    text_features = FeatureUnion([
        ('word', CountVectorizer(ngram_range=(1, 2), max_features=5000,
                                 preprocessor=lambda x: '' if pd.isna(x) else str(x),
                                 lowercase=True)),
        ('char', CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000,
                                 preprocessor=lambda x: '' if pd.isna(x) else str(x),
                                 lowercase=True)),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                                  preprocessor=lambda x: '' if pd.isna(x) else str(x),
                                  lowercase=True)),
    ])

    dict_features = Pipeline([
        ('extract', DictFeatureTransformer()),
        ('vectorize', DictVectorizer(sparse=False))
    ])

    return Pipeline([
            ('features', FeatureUnion([
                ('text', text_features),
                ('dict', dict_features)
            ])),
            ('scaler', MaxAbsScaler()),
            ('abs', AbsoluteTransformer()),
            ('classifier', MultinomialNB())
        ])

def combine_datasets(unredactor_df):
    if os.path.exists(IMDB_NAMES_FILE):
        print("Loading IMDB names data...")
        imdb_df = pd.read_csv(IMDB_NAMES_FILE)
        imdb_df = imdb_df.dropna(subset=['name'])
        imdb_processed = pd.DataFrame({
            'split': ['training'] * len(imdb_df),
            'name': imdb_df['name'].astype(str),
            'context': imdb_df['name'].astype(str).apply(lambda x: '█' * len(x))
        })
        imdb_processed = imdb_processed.drop_duplicates(subset=['name'])
        combined_df = pd.concat([unredactor_df, imdb_processed], ignore_index=True)
        print(f"Added {len(imdb_processed)} unique names from IMDB dataset")
        return combined_df
    else:
        print(f"Warning: IMDB names file {IMDB_NAMES_FILE} not found. Using only unredactor.tsv data.")
        return unredactor_df

def train_model(df, pipeline):
    df = df.copy()
    df['context'] = df['context'].fillna('').astype(str)
    df['name'] = df['name'].fillna('').astype(str)
    train_mask = df['split'] == 'training'
    X = df[train_mask]['context'].tolist()
    y = df[train_mask]['name'].values
    
    # Remove classes with only one instance
    class_counts = Counter(y)
    valid_classes = {cls for cls, count in class_counts.items() if count > 1}
    mask = np.isin(y, list(valid_classes))
    X = [x for x, m in zip(X, mask) if m]
    y = y[mask]
    
    print(f"Training on {len(X)} samples with {len(valid_classes)} unique classes")
    pipeline.fit(X, y)
    return pipeline

def evaluate_model(df, pipeline):
    df = df.copy()
    df['context'] = df['context'].fillna('').astype(str)
    df['name'] = df['name'].fillna('').astype(str)
    val_mask = df['split'] == 'validation'
    X_val = df[val_mask]['context'].tolist()
    y_val = df[val_mask]['name'].values
    
    # Use the same classes as in training
    valid_classes = set(pipeline.named_steps['classifier'].classes_)
    mask = np.isin(y_val, list(valid_classes))
    X_val = [x for x, m in zip(X_val, mask) if m]
    y_val = y_val[mask]
    
    print(f"Evaluating on {len(X_val)} samples")
    y_pred = pipeline.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted', zero_division=0)
    return precision, recall, f1

def unredact(context, pipeline):
    processed_context = preprocess(context)
    return pipeline.predict([str(processed_context)])[0]

def main(train_file, test_file=None, submission_file=None):
    print("Loading training data...")
    try:
        unredactor_df = pd.read_csv(train_file, sep='\t', names=['split', 'name', 'context'], 
                                  on_bad_lines='skip', quoting=3)
        print("Successfully loaded training data with skipped bad lines")
    except Exception as e:
        print(f"Error during file loading: {str(e)}")
        return
    
    combined_df = combine_datasets(unredactor_df)
    
    print(f"Total examples: {len(combined_df)}")
    print(f"Training examples: {len(combined_df[combined_df['split'] == 'training'])}")
    print(f"Validation examples: {len(combined_df[combined_df['split'] == 'validation'])}")

    print("Training MultinomialNB model...")
    pipeline = create_pipeline()
    trained_pipeline = train_model(combined_df, pipeline)

    print("Evaluating model...")
    precision, recall, f1 = evaluate_model(combined_df, trained_pipeline)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

    if test_file and submission_file:
        print("Processing test file...")
        test_df = pd.read_csv(test_file, sep='\t', names=['id','context'], 
                                  on_bad_lines='skip', quoting=3)
        test_df['name'] = test_df['context'].apply(lambda x: unredact(x, trained_pipeline))
        #test_df = test_df.drop_duplicates(subset=["name"])
        test_df[[ 'id','name']].to_csv(submission_file, sep='\t', index=False)
        print(f"Submission file created: {submission_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unredactor using MultinomialNB")
    parser.add_argument("train_file", help="Path to the training file (unredactor.tsv)")
    parser.add_argument("--test_file", help="Path to the test file", default=None)
    parser.add_argument("--submission_file", help="Path to the output submission file", default=None)
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.submission_file)
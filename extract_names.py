import pandas as pd
import spacy
import os
from tqdm import tqdm

# Static path configuration
IMDB_DIR = "./aclImdb"
NAMES_OUTPUT_FILE = "imdb_names.csv"

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def process_directory_for_names(directory_path, sentiment, split):
    """Process all files in a directory and extract names."""
    names_data = []
    
    txt_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    for file_path in tqdm(txt_files, desc=f"Processing {split}/{sentiment}"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                review = f.read()
                doc = nlp(review)
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        redacted_review = review.replace(ent.text, '█' * len(ent.text))
                        names_data.append({
                            'name': ent.text,
                            'sentiment': sentiment,
                            'split': split,
                            'context': redacted_review,
                            'original_file': file_path
                        })
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    return names_data

def main():
    all_data = []
    
    # Define the structure based on the directory layout
    splits = {
        'train': ['pos', 'neg'],
        'test': ['pos', 'neg']
    }
    
    for split, sentiments in splits.items():
        for sentiment in sentiments:
            dir_path = os.path.join(IMDB_DIR, split, sentiment)
            if os.path.exists(dir_path):
                data = process_directory_for_names(dir_path, sentiment, split)
                all_data.extend(data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save names to CSV
    names_df = df[['name']].drop_duplicates()
    names_df.to_csv(NAMES_OUTPUT_FILE, index=False)
    print(f"\nExtracted {len(names_df)} unique names")
    print(f"Names saved to {NAMES_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
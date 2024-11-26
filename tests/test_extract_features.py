import pytest
from unredactor import extract_features

def test_extract_features_basic():
    text = "This is a test. ███████ was redacted."
    features = extract_features(text)
    assert features['num_words'] == 6, f"Expected 6 words, got {features['num_words']}"
    assert features['num_redactions'] == 1, f"Expected 1 redaction, got {features['num_redactions']}"
    assert features['redaction_length'] == 7, f"Expected redaction length 7, got {features['redaction_length']}"
    assert features['num_sentences'] == 2, f"Expected 2 sentences, got {features['num_sentences']}"
    assert features['avg_word_length'] > 3, f"Expected average word length > 3, got {features['avg_word_length']}"
    assert features['lexical_diversity'] > 0.6, f"Expected lexical diversity > 0.6, got {features['lexical_diversity']}"
    
def test_extract_features_empty():
    text = ""
    features = extract_features(text)
    for key in features:
        assert features[key] == 0, f"Feature {key} should be 0 for empty text"

def test_extract_features_special_characters():
    text = "Redacted █████. Symbols: @#$%^&*() are present."
    features = extract_features(text)
    assert features['num_words'] == 4, f"Expected 4 words, got {features['num_words']}"
    assert features['num_special_chars'] == 17, f"Expected 17 special characters, got {features['num_special_chars']}"

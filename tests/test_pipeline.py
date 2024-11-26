import pytest
from sklearn.pipeline import Pipeline
from unredactor import create_pipeline

def test_create_pipeline():
    pipeline = create_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert 'features' in pipeline.named_steps
    assert 'scaler' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps

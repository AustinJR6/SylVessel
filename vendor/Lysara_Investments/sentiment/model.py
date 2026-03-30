from __future__ import annotations

"""Sentiment model utilities."""

from functools import lru_cache
from typing import Iterable, List, Dict
from transformers import pipeline

# Default HuggingFace model for financial sentiment
DEFAULT_MODEL = "ProsusAI/finbert"

@lru_cache()
def get_sentiment_pipeline(model_name: str = DEFAULT_MODEL):
    """Return a cached sentiment-analysis pipeline."""
    return pipeline("sentiment-analysis", model=model_name)


def analyze_texts(texts: Iterable[str], model_name: str = DEFAULT_MODEL) -> List[Dict[str, float]]:
    """Run sentiment analysis on a list of texts."""
    nlp = get_sentiment_pipeline(model_name)
    clean_texts = [t.replace("\n", " ").strip() for t in texts if t and t.strip()]
    if not clean_texts:
        return []
    return nlp(clean_texts)


def label_to_score(label: str, score: float) -> float:
    """Convert a model label and confidence to a [-1, 1] score."""
    label = label.lower()
    if label == "positive":
        val = score
    elif label == "negative":
        val = -score
    else:
        val = 0.0
    return val

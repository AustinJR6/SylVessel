"""Multilingual sentiment scraping support."""
from __future__ import annotations

from typing import List, Dict

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - transformers optional
    pipeline = None


def analyze_texts(texts: List[str], lang: str = "en") -> List[Dict]:
    """Return sentiment labels using a multilingual model."""
    if pipeline is None:
        return [{"label": "neutral", "score": 0.0} for _ in texts]
    model = pipeline("sentiment-analysis", model="xlm-roberta-base")
    return [dict(r) for r in model(texts)]


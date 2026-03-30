"""News headline sentiment via GNews or NewsAPI."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict
import os

try:
    from gnews import GNews
except Exception:  # library not installed
    GNews = None

try:
    from newsapi import NewsApiClient
except Exception:
    NewsApiClient = None

from .model import analyze_texts, label_to_score


class NewsSentiment:
    """Fetch news articles and run sentiment analysis."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("NEWSAPI_KEY")
        if self.api_key and NewsApiClient:
            self.client = NewsApiClient(api_key=self.api_key)
        else:
            self.client = None
        self.gnews = GNews() if GNews else None

    def fetch_headlines(self, ticker: str, limit: int = 10) -> List[str]:
        texts: List[str] = []
        if self.client:
            data = self.client.get_everything(q=ticker, language="en", sort_by="publishedAt", page_size=limit)
            texts.extend([f"{a.get('title', '')} {a.get('description', '')}" for a in data.get('articles', [])])
        elif self.gnews:
            results = self.gnews.get_news(ticker)[:limit]
            texts.extend([f"{a.get('title', '')} {a.get('description', '')}" for a in results])
        return texts

    def analyze(self, ticker: str, limit: int = 10) -> Dict:
        texts = self.fetch_headlines(ticker, limit)
        results = analyze_texts(texts)
        scores = [label_to_score(r['label'], r['score']) for r in results]
        avg = sum(scores) / len(scores) if scores else 0.0
        label = "neutral"
        if avg > 0.1:
            label = "positive"
        elif avg < -0.1:
            label = "negative"
        return {
            "score": round((avg + 1) / 2, 3),
            "label": label,
            "count": len(scores),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

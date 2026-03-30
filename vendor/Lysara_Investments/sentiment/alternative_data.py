"""Alternative data sources for sentiment."""
from __future__ import annotations

from typing import Dict, List

# TODO: integrate actual APIs for these sources

def youtube_sentiment(video_id: str) -> Dict:
    """Placeholder for YouTube transcript sentiment."""
    return {"video": video_id, "score": 0.0, "label": "neutral"}


def google_trends(keyword: str) -> List[int]:
    """Placeholder for Google Trends interest."""
    return []


def earnings_call_sentiment(ticker: str) -> Dict:
    """Placeholder for earnings call transcript sentiment."""
    return {"ticker": ticker, "score": 0.0, "label": "neutral"}


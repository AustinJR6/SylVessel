"""Unified sentiment API wrapper."""
from __future__ import annotations

from typing import Dict

from .sentiment_handler import SentimentHandler
from .volatility_detector import compute_volatility
from .feedback_loop import log_trade_result


handler = SentimentHandler()


def get_sentiment_score(ticker: str) -> Dict:
    """Return structured sentiment information for a ticker."""
    result = handler.get_sentiment_score(ticker)
    volatility = compute_volatility({"score": result["score"]})
    result["volatility"] = volatility
    return result


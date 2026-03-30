"""Aggregate sentiment from multiple sources into a Sentiment Confidence Score."""
from __future__ import annotations

from typing import Dict
from datetime import datetime, timezone

from .reddit_sentiment import RedditSentiment
from .twitter_sentiment import analyze_twitter_sentiment
from .news_sentiment import NewsSentiment


class SentimentHandler:
    """Interface used by strategies and services to obtain a sentiment score."""

    def __init__(self, weights: Dict[str, float] | None = None, subreddits: list[str] | None = None, news_api_key: str | None = None):
        self.weights = weights or {"twitter": 0.3, "reddit": 0.2, "news": 0.5}
        self.subreddits = subreddits or ["stocks", "cryptocurrency"]
        self.reddit = RedditSentiment()
        self.news = NewsSentiment(api_key=news_api_key)

    def _combine(self, results: Dict[str, Dict]) -> Dict:
        """Combine individual source scores into a final SCS."""
        total = 0.0
        weight_sum = 0.0
        for src, res in results.items():
            w = self.weights.get(src, 0)
            count = res.get("count", 1)
            w = w * min(1.0, count / 10.0)  # volume adjustment
            total += res.get("score", 0) * w
            weight_sum += w
        score = total / weight_sum if weight_sum else 0.0
        label = "neutral"
        if score > 0.6:
            label = "positive"
        elif score < 0.4:
            label = "negative"
        return {"score": round(score, 3), "label": label, "sources": results}

    def get_sentiment_score(self, ticker: str) -> Dict:
        """Public method to compute the sentiment score for a ticker."""
        results: Dict[str, Dict] = {}
        results["twitter"] = analyze_twitter_sentiment(ticker)
        reddit_scores = []
        for sub in self.subreddits:
            reddit_scores.append(self.reddit.analyze(sub, ticker))
        if reddit_scores:
            avg_score = sum(r["score"] for r in reddit_scores) / len(reddit_scores)
            avg_count = sum(r["count"] for r in reddit_scores)
            results["reddit"] = {
                "score": avg_score,
                "count": avg_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "label": "positive" if avg_score > 0.6 else "negative" if avg_score < 0.4 else "neutral",
            }
        results["news"] = self.news.analyze(ticker)
        return self._combine(results)


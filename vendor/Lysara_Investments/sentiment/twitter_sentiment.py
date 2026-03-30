"""Twitter sentiment scraping using snscrape."""
from __future__ import annotations

from datetime import datetime, timezone

import snscrape.modules.twitter as sntwitter

from .model import analyze_texts, label_to_score


def fetch_tweets(ticker: str, limit: int = 50) -> List[str]:
    """Collect recent tweets mentioning the given ticker."""
    query = f"{ticker}"
    texts: List[str] = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        texts.append(tweet.content)
        if i + 1 >= limit:
            break
    return texts


def analyze_twitter_sentiment(ticker: str, limit: int = 50) -> Dict:
    tweets = fetch_tweets(ticker, limit)
    results = analyze_texts(tweets)
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

"""Reddit sentiment scraping using PRAW."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict
import os
import logging
import praw

from .model import analyze_texts, label_to_score


class RedditSentiment:
    """Fetch posts and comments from a subreddit and run sentiment analysis."""

    def __init__(self, client_id: str | None = None, client_secret: str | None = None, user_agent: str | None = None):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.getenv("REDDIT_USER_AGENT", "LysaraSentimentBot")
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            check_for_async=False,
        )

    def fetch_texts(self, subreddit: str, ticker: str, limit: int = 20) -> List[str]:
        """Gather submission titles and comments mentioning the ticker."""
        texts: List[str] = []
        query = ticker.upper()
        for submission in self.reddit.subreddit(subreddit).search(query, limit=limit, sort="top", time_filter="day"):
            texts.append(f"{submission.title} {submission.selftext}")
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if query in comment.body.upper():
                    texts.append(comment.body)
        return texts

    def analyze(self, subreddit: str, ticker: str, limit: int = 20) -> Dict:
        texts = self.fetch_texts(subreddit, ticker, limit)
        results = analyze_texts(texts)
        scores = [label_to_score(r['label'], r['score']) for r in results]
        avg = sum(scores) / len(scores) if scores else 0.0
        label = "neutral"
        if avg > 0.1:
            label = "positive"
        elif avg < -0.1:
            label = "negative"
        return {
            "score": round((avg + 1) / 2, 3),  # convert to 0-1 scale
            "label": label,
            "count": len(scores),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

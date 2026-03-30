# data/sentiment.py

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote

import aiohttp
from textblob import TextBlob


_CRYPTO_DISPLAY_NAMES = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "SOL": "Solana",
    "ADA": "Cardano",
    "XRP": "XRP",
    "BNB": "BNB",
    "DOT": "Polkadot",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def analyze_sentiment(text: str) -> float:
    """
    Returns a polarity score between -1.0 (negative) and 1.0 (positive).
    """
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        logging.warning(f"Sentiment analysis failed: {e}")
        return 0.0


def symbol_base_asset(symbol: str) -> str:
    token = str(symbol or "").strip().upper()
    if "-" in token:
        token = token.split("-", 1)[0]
    return token


def symbol_display_name(symbol: str) -> str:
    asset = symbol_base_asset(symbol)
    return _CRYPTO_DISPLAY_NAMES.get(asset, asset.title())


def symbol_aliases(symbol: str) -> list[str]:
    asset = symbol_base_asset(symbol)
    display = symbol_display_name(symbol)
    aliases = [asset]
    if display.upper() != asset:
        aliases.append(display)
    if asset == "BTC":
        aliases.append("BTC-USD")
    elif asset == "ETH":
        aliases.append("ETH-USD")
    elif asset == "SOL":
        aliases.append("SOL-USD")
    elif asset == "ADA":
        aliases.append("ADA-USD")
    return aliases


def news_query_for_symbol(symbol: str) -> str:
    aliases = symbol_aliases(symbol)
    if len(aliases) == 1:
        return aliases[0]
    return " OR ".join(f'"{alias}"' if " " in alias else alias for alias in aliases[:3])


def reddit_query_for_symbol(symbol: str) -> str:
    aliases = symbol_aliases(symbol)
    return " OR ".join(aliases[:3])


def x_query_for_symbol(symbol: str) -> str:
    asset = symbol_base_asset(symbol)
    aliases = symbol_aliases(symbol)
    tokens = [asset, f"#{asset}", f"${asset}"]
    if len(aliases) > 1:
        tokens.append(f'"{aliases[1]}"')
    query = " OR ".join(dict.fromkeys(tokens))
    return f"({query}) lang:en -is:retweet"


def _result_payload(*, source: str, score: float, count: int, query: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "source": source,
        "score": round(float(score), 3),
        "count": int(max(count, 0)),
        "timestamp": _utcnow_iso(),
    }
    if query:
        payload["query"] = query
    if extra:
        payload.update(extra)
    return payload


def _mean_score(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


async def fetch_cryptopanic_sentiment(api_key: str, symbols: list[str], base_url: str | None = None) -> dict[str, Any]:
    """Pull latest crypto news from CryptoPanic and score sentiment per symbol."""
    if not api_key:
        return {}
    root = (base_url or "https://cryptopanic.com/api/developer/v2").rstrip("/")
    url = f"{root}/posts/"
    headers = {"Accept": "application/json"}
    result: dict[str, Any] = {}
    async with aiohttp.ClientSession() as session:
        for symbol in symbols:
            asset = symbol_base_asset(symbol).lower()
            params = {
                "auth_token": api_key,
                "currencies": asset,
                "kind": "news",
                "public": "true",
            }
            try:
                async with session.get(url, params=params, headers=headers) as response:
                    data = await response.json(content_type=None)
                    posts = data.get("results", [])
                    scores = [
                        analyze_sentiment((post.get("title") or "") + " " + (post.get("body") or ""))
                        for post in posts
                    ]
                    result[symbol.upper()] = _result_payload(
                        source="cryptopanic",
                        score=_mean_score(scores),
                        count=len(scores),
                        query=asset,
                    )
            except Exception as e:
                logging.error(f"CryptoPanic error for {symbol}: {e}")
                result[symbol.upper()] = _result_payload(source="cryptopanic", score=0.0, count=0, query=asset)
    return result


async def fetch_newsapi_sentiment(api_key: str, query: str = "Bitcoin", limit: int = 25) -> dict[str, Any]:
    """
    Fetch recent headlines from NewsAPI and perform sentiment analysis.
    """
    if not api_key:
        return _result_payload(source="newsapi", score=0.0, count=0, query=query)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max(5, min(int(limit or 25), 100)),
        "apiKey": api_key,
    }
    scores: list[float] = []
    titles: list[str] = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json(content_type=None)
                articles = data.get("articles", [])
                for article in articles:
                    title = article.get("title") or ""
                    desc = article.get("description") or ""
                    content = (title + " " + desc).strip()
                    if not content:
                        continue
                    titles.append(title)
                    scores.append(analyze_sentiment(content))
    except Exception as e:
        logging.error(f"NewsAPI error for query {query}: {e}")
    return _result_payload(
        source="newsapi",
        score=_mean_score(scores),
        count=len(scores),
        query=query,
        extra={"titles": titles[:5]},
    )


async def _reddit_access_token(
    session: aiohttp.ClientSession,
    *,
    client_id: str,
    client_secret: str,
    user_agent: str,
) -> str | None:
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {auth}",
        "User-Agent": user_agent,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {"grant_type": "client_credentials"}
    try:
        async with session.post("https://www.reddit.com/api/v1/access_token", data=payload, headers=headers) as response:
            data = await response.json(content_type=None)
            return str(data.get("access_token") or "").strip() or None
    except Exception as exc:
        logging.error("Reddit auth error: %s", exc)
        return None


async def fetch_reddit_sentiment(
    subreddit: str,
    limit: int = 50,
    *,
    query: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    user_agent: str = "LysaraSentimentBot/1.0",
) -> dict[str, Any]:
    """Gather sentiment score from a subreddit or search result set."""
    subreddit = str(subreddit or "").strip()
    if not subreddit:
        return _result_payload(source="reddit", score=0.0, count=0, query=query or "")

    public_headers = {"User-Agent": user_agent}
    scores: list[float] = []
    titles: list[str] = []
    effective_query = (query or "").strip()

    try:
        async with aiohttp.ClientSession() as session:
            oauth_token = None
            if client_id and client_secret:
                oauth_token = await _reddit_access_token(
                    session,
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                )

            if effective_query:
                url = (
                    f"https://oauth.reddit.com/r/{subreddit}/search"
                    if oauth_token
                    else f"https://www.reddit.com/r/{subreddit}/search.json"
                )
                params = {
                    "q": effective_query,
                    "restrict_sr": "1",
                    "sort": "new",
                    "limit": max(5, min(int(limit or 25), 100)),
                    "t": "day",
                }
            else:
                url = (
                    f"https://oauth.reddit.com/r/{subreddit}/hot"
                    if oauth_token
                    else f"https://www.reddit.com/r/{subreddit}/hot.json"
                )
                params = {"limit": max(5, min(int(limit or 25), 100))}

            headers = dict(public_headers)
            if oauth_token:
                headers["Authorization"] = f"Bearer {oauth_token}"

            async with session.get(url, params=params, headers=headers) as response:
                data = await response.json(content_type=None)
                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    post_data = post.get("data", {})
                    title = post_data.get("title", "")
                    text = (title + " " + (post_data.get("selftext", "") or "")).strip()
                    if not text:
                        continue
                    titles.append(title)
                    scores.append(analyze_sentiment(text))
    except Exception as e:
        logging.error(f"Reddit sentiment error for r/{subreddit}: {e}")
    return _result_payload(
        source="reddit",
        score=_mean_score(scores),
        count=len(scores),
        query=effective_query or subreddit,
        extra={"subreddit": subreddit, "titles": titles[:5]},
    )


async def fetch_x_sentiment(bearer_token: str, symbol: str, limit: int = 25) -> dict[str, Any]:
    """Fetch recent X posts for a symbol via the official recent search API."""
    token = unquote(str(bearer_token or "").strip())
    if not token:
        return _result_payload(source="x", score=0.0, count=0, query=x_query_for_symbol(symbol))

    query = x_query_for_symbol(symbol)
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": max(10, min(int(limit or 25), 100)),
        "tweet.fields": "created_at,lang,public_metrics",
    }
    headers = {"Authorization": f"Bearer {token}"}
    texts: list[str] = []

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                data = await response.json(content_type=None)
                tweets = data.get("data", [])
                for tweet in tweets:
                    text = str(tweet.get("text") or "").strip()
                    if text:
                        texts.append(text)
    except Exception as exc:
        logging.error("X sentiment error for %s: %s", symbol, exc)

    scores = [analyze_sentiment(text) for text in texts]
    return _result_payload(
        source="x",
        score=_mean_score(scores),
        count=len(scores),
        query=query,
    )

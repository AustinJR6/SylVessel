import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List

from indicators.technical_indicators import (
    exponential_moving_average,
    relative_strength_index,
    bollinger_bands,
)

from data.sentiment import (
    fetch_cryptopanic_sentiment,
    fetch_newsapi_sentiment,
    fetch_reddit_sentiment,
)

try:
    from pytrends.request import TrendReq
except Exception:  # pytrends optional
    TrendReq = None

@dataclass
class FusionResult:
    symbol: str
    conviction: float
    details: Dict[str, float]

class SignalFusionEngine:
    """Fuse technical, sentiment and market data signals into a conviction score."""

    def __init__(self, config: Dict[str, float]):
        self.tech_weight = float(config.get("TECH_WEIGHT", 0.5))
        self.sent_weight = float(config.get("SENT_WEIGHT", 0.3))
        self.market_weight = float(config.get("MARKET_WEIGHT", 0.2))
        self.reddit_subs = config.get("REDDIT_SUBS", ["CryptoCurrency"])
        self.news_key = config.get("NEWSAPI_KEY")
        self.cryptopanic_key = config.get("api_keys", {}).get("cryptopanic")
        self.cryptopanic_base_url = config.get("api_keys", {}).get("cryptopanic_base_url")
        self.loop = asyncio.get_event_loop()

    async def sentiment_score(self, symbol: str) -> float:
        """Aggregate sentiment from multiple sources."""
        tasks = []
        if self.news_key:
            tasks.append(fetch_newsapi_sentiment(self.news_key, symbol))
        if self.cryptopanic_key:
            tasks.append(fetch_cryptopanic_sentiment(self.cryptopanic_key, [symbol], base_url=self.cryptopanic_base_url))
        for sub in self.reddit_subs:
            tasks.append(fetch_reddit_sentiment(sub))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        scores = []
        for r in results:
            if isinstance(r, dict):
                if "score" in r:
                    scores.append(r.get("score", 0.0))
                else:
                    scores.extend([v.get("score", 0.0) for v in r.values() if isinstance(v, dict)])
        return sum(scores) / max(len(scores), 1)

    async def google_trend_score(self, keyword: str) -> float:
        if not TrendReq:
            return 0.0
        try:
            pt = TrendReq(hl="en-US", timeout=(10, 25))
            kw = [keyword]
            pt.build_payload(kw_list=kw, timeframe="now 7-d")
            data = pt.interest_over_time()
            if data.empty:
                return 0.0
            val = float(data[keyword].iloc[-1])
            return val / 100.0
        except Exception as e:
            logging.error(f"Pytrends failed: {e}")
            return 0.0

    def technical_score(self, prices: List[float]) -> float:
        if len(prices) < 5:
            return 0.5
        ema = exponential_moving_average(prices, 10)
        rsi = relative_strength_index(prices)
        upper, lower = bollinger_bands(prices)
        price = prices[-1]
        tech = 0.5
        if price > ema and rsi > 50:
            tech += 0.25
        if upper and price > upper:
            tech += 0.25
        if lower and price < lower:
            tech -= 0.25
        tech = min(max(tech, 0), 1)
        return tech

    def market_score(self, prices: List[float]) -> float:
        if len(prices) < 2:
            return 0.5
        momentum = (prices[-1] - prices[0]) / abs(prices[0])
        volatility = max(prices) - min(prices)
        score = 0.5 + momentum - (volatility / prices[-1])
        return min(max(score, 0), 1)

    async def score_symbol(self, symbol: str, prices: List[float]) -> FusionResult:
        tech = self.technical_score(prices)
        sent = await self.sentiment_score(symbol)
        trend = await self.google_trend_score(symbol.split("-")[0])
        market = self.market_score(prices)
        conviction = (
            tech * self.tech_weight
            + ((sent + trend) / 2) * self.sent_weight
            + market * self.market_weight
        )
        conviction = round(min(max(conviction, 0.0), 1.0), 3)
        details = {
            "tech": round(tech, 3),
            "sent": round(sent, 3),
            "trend": round(trend, 3),
            "market": round(market, 3),
        }
        return FusionResult(symbol=symbol, conviction=conviction, details=details)


# services/background_tasks.py

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from data.event_risk import (
    fetch_coinmarketcal_events,
    fetch_finnhub_economic_calendar,
    fetch_tradingeconomics_calendar,
)
from data.sentiment import (
    fetch_cryptopanic_sentiment,
    fetch_newsapi_sentiment,
    fetch_reddit_sentiment,
    fetch_x_sentiment,
    news_query_for_symbol,
    reddit_query_for_symbol,
    symbol_base_asset,
    symbol_display_name,
)
from services.daemon_state import get_state
from services.event_risk_service import EventRiskService


class BackgroundTasks:
    def __init__(self, config: dict):
        self.config = config
        self.crypto_symbols = [str(symbol).upper() for symbol in config.get("TRADE_SYMBOLS", ["BTC-USD", "ETH-USD"])]
        api_keys = config.get("api_keys", {})
        self.newsapi_key = api_keys.get("newsapi")
        self.cryptopanic_key = api_keys.get("cryptopanic")
        self.cryptopanic_base_url = api_keys.get("cryptopanic_base_url")
        self.reddit_client_id = api_keys.get("reddit_client_id")
        self.reddit_client_secret = api_keys.get("reddit_secret")
        self.reddit_user_agent = api_keys.get("reddit_user_agent", "LysaraSentimentBot/1.0")
        self.x_bearer_token = api_keys.get("x_bearer_token")
        self.finnhub_api_key = api_keys.get("finnhub_api_key")
        self.tradingeconomics_api_key = api_keys.get("tradingeconomics_api_key")
        self.tradingeconomics_api_secret = api_keys.get("tradingeconomics_api_secret")
        self.coinmarketcal_api_key = api_keys.get("coinmarketcal_api_key")
        self.subreddits = config.get("reddit_subreddits", ["Cryptocurrency"])
        self.sentiment_scores: dict[str, Any] = {}
        self._running = True
        self.sentiment_file = Path("dashboard/data/sentiment_cache.json")
        self.event_risk_service = EventRiskService(config=config)
        self.event_risk_file = self.event_risk_service.file_path

    def _persist_scores(self):
        try:
            self.sentiment_file.parent.mkdir(parents=True, exist_ok=True)
            self.sentiment_file.write_text(json.dumps(self.sentiment_scores, indent=2))
        except Exception as e:
            logging.error(f"Failed to persist sentiment scores: {e}")

    def _persist_event_risk(self, payload: dict[str, Any]):
        try:
            self.event_risk_service.persist_snapshot(payload)
        except Exception as e:
            logging.error(f"Failed to persist event risk snapshot: {e}")

    async def _collect_newsapi_scores(self) -> dict[str, Any]:
        if not self.newsapi_key or not self.crypto_symbols:
            return {}
        tasks = [
            fetch_newsapi_sentiment(self.newsapi_key, query=news_query_for_symbol(symbol), limit=20)
            for symbol in self.crypto_symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        payload: dict[str, Any] = {}
        for symbol, result in zip(self.crypto_symbols, results):
            if isinstance(result, Exception):
                logging.error("NewsAPI sentiment failed for %s: %s", symbol, result)
                continue
            payload[symbol] = result
        return payload

    async def _collect_reddit_scores(self) -> dict[str, Any]:
        if not self.subreddits or not self.crypto_symbols:
            return {}
        payload: dict[str, Any] = {}
        for symbol in self.crypto_symbols:
            tasks = [
                fetch_reddit_sentiment(
                    subreddit=subreddit,
                    limit=20,
                    query=reddit_query_for_symbol(symbol),
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent,
                )
                for subreddit in self.subreddits
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            entries = [result for result in results if isinstance(result, dict)]
            payload[symbol] = self._combine_source_entries(
                source="reddit",
                symbol=symbol,
                entries=entries,
                query=reddit_query_for_symbol(symbol),
            )
        return payload

    async def _collect_x_scores(self) -> dict[str, Any]:
        if not self.x_bearer_token or not self.crypto_symbols:
            return {}
        tasks = [fetch_x_sentiment(self.x_bearer_token, symbol, limit=25) for symbol in self.crypto_symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        payload: dict[str, Any] = {}
        for symbol, result in zip(self.crypto_symbols, results):
            if isinstance(result, Exception):
                logging.error("X sentiment failed for %s: %s", symbol, result)
                continue
            payload[symbol] = result
        return payload

    async def _collect_macro_events(self) -> tuple[list[dict[str, Any]], list[str]]:
        now = datetime.now(timezone.utc)
        start_date = now.date().isoformat()
        end_date = (now + timedelta(hours=max(24, int(self.config.get("event_risk_lookahead_hours", 24)) + 6))).date().isoformat()
        events: list[dict[str, Any]] = []
        providers: list[str] = []

        if self.finnhub_api_key:
            finnhub_events = await fetch_finnhub_economic_calendar(self.finnhub_api_key, start_date, end_date)
            events.extend(finnhub_events)
            providers.append("finnhub")

        if self.tradingeconomics_api_key and self.tradingeconomics_api_secret:
            te_events = await fetch_tradingeconomics_calendar(
                self.tradingeconomics_api_key,
                self.tradingeconomics_api_secret,
                start_date,
                end_date,
            )
            events.extend(te_events)
            providers.append("tradingeconomics")

        return events, providers

    async def _collect_crypto_calendar_events(self) -> tuple[list[dict[str, Any]], list[str]]:
        assets = [symbol_base_asset(symbol) for symbol in self.crypto_symbols]
        events: list[dict[str, Any]] = []
        providers: list[str] = []
        if self.coinmarketcal_api_key and assets:
            coinmarketcal_events = await fetch_coinmarketcal_events(self.coinmarketcal_api_key, assets)
            events.extend(coinmarketcal_events)
            providers.append("coinmarketcal")
        return events, providers

    def _combine_source_entries(self, *, source: str, symbol: str, entries: list[dict[str, Any]], query: str) -> dict[str, Any]:
        valid = [entry for entry in entries if isinstance(entry, dict)]
        if not valid:
            return {
                "source": source,
                "score": 0.0,
                "count": 0,
                "timestamp": None,
                "query": query,
            }
        total_weight = 0.0
        weighted_score = 0.0
        total_count = 0
        timestamps: list[str] = []
        titles: list[str] = []
        subreddits: list[str] = []
        for entry in valid:
            count = int(entry.get("count") or 0)
            score = float(entry.get("score") or 0.0)
            weight = max(count, 1)
            total_weight += weight
            weighted_score += score * weight
            total_count += count
            if entry.get("timestamp"):
                timestamps.append(str(entry["timestamp"]))
            titles.extend([str(title) for title in (entry.get("titles") or []) if str(title).strip()])
            if entry.get("subreddit"):
                subreddits.append(str(entry["subreddit"]))
        payload = {
            "source": source,
            "score": round(weighted_score / total_weight, 3) if total_weight > 0 else 0.0,
            "count": total_count,
            "timestamp": max(timestamps) if timestamps else None,
            "query": query,
        }
        if titles:
            payload["titles"] = titles[:6]
        if subreddits:
            payload["subreddits"] = sorted(dict.fromkeys(subreddits))
        return payload

    def _build_symbol_entry(self, symbol: str, configured_sources: list[str], source_maps: dict[str, dict[str, Any]]) -> dict[str, Any]:
        sources: dict[str, dict[str, Any]] = {}
        total_mentions = 0
        weighted_score = 0.0
        total_weight = 0.0
        raw_scores: list[float] = []
        updated_at: str | None = None

        for source_name, source_map in source_maps.items():
            entry = (source_map or {}).get(symbol)
            if not isinstance(entry, dict):
                continue
            count = int(entry.get("count") or 0)
            if count <= 0:
                continue
            score = float(entry.get("score") or 0.0)
            weight = max(count, 1)
            total_mentions += count
            total_weight += weight
            weighted_score += score * weight
            raw_scores.append(score)
            sources[source_name] = entry
            candidate_ts = str(entry.get("timestamp") or "").strip()
            if candidate_ts and (updated_at is None or candidate_ts > updated_at):
                updated_at = candidate_ts

        source_count = len(sources)
        composite_score = round(weighted_score / total_weight, 3) if total_weight > 0 else 0.0
        score_spread = (max(raw_scores) - min(raw_scores)) if len(raw_scores) > 1 else 0.0
        coverage = round(source_count / max(len(configured_sources), 1), 3) if configured_sources else 0.0
        confidence = 0.2
        confidence += min(source_count, 4) * 0.15
        confidence += min(total_mentions, 60) / 100.0
        confidence -= min(score_spread, 1.0) * 0.2
        confidence = round(max(0.05, min(confidence, 0.95)), 3)

        anomaly_flags: list[str] = []
        if source_count < 2:
            anomaly_flags.append("thin_coverage")
        if score_spread >= 0.45:
            anomaly_flags.append("source_divergence")
        if total_mentions >= 75:
            anomaly_flags.append("attention_spike")

        return {
            "symbol": symbol,
            "asset": symbol_base_asset(symbol),
            "display_name": symbol_display_name(symbol),
            "sources": sources,
            "composite": {
                "score": composite_score,
                "mention_velocity": total_mentions,
                "source_count": source_count,
                "source_coverage": coverage,
                "confidence": confidence,
                "anomaly_flags": anomaly_flags,
            },
            "updated_at": updated_at,
        }

    async def run_sentiment_loop(self, interval: int | None = None):
        """
        Loop fetching symbol-centric sentiment snapshots at a defined interval.
        """
        loop_interval = max(60, int(interval if interval is not None else self.config.get("sentiment_loop_interval_seconds", 300)))
        while self._running:
            logging.info("Running sentiment fetch loop...")
            configured_sources: list[str] = []

            news_by_symbol = await self._collect_newsapi_scores()
            if self.newsapi_key:
                configured_sources.append("newsapi")
                logging.info("NewsAPI symbol sentiment updated for %s", ", ".join(news_by_symbol.keys()) or "none")

            cryptopanic_by_symbol = {}
            if self.cryptopanic_key and self.crypto_symbols:
                cryptopanic_by_symbol = await fetch_cryptopanic_sentiment(
                    self.cryptopanic_key,
                    self.crypto_symbols,
                    base_url=self.cryptopanic_base_url,
                )
                configured_sources.append("cryptopanic")
                logging.info("CryptoPanic symbol sentiment updated for %s", ", ".join(cryptopanic_by_symbol.keys()) or "none")

            reddit_by_symbol = await self._collect_reddit_scores()
            if self.subreddits:
                configured_sources.append("reddit")
                logging.info("Reddit symbol sentiment updated for %s", ", ".join(reddit_by_symbol.keys()) or "none")

            x_by_symbol = await self._collect_x_scores()
            if self.x_bearer_token:
                configured_sources.append("x")
                logging.info("X symbol sentiment updated for %s", ", ".join(x_by_symbol.keys()) or "none")

            symbol_entries = {
                symbol: self._build_symbol_entry(
                    symbol,
                    configured_sources=configured_sources,
                    source_maps={
                        "newsapi": news_by_symbol,
                        "cryptopanic": cryptopanic_by_symbol,
                        "reddit": reddit_by_symbol,
                        "x": x_by_symbol,
                    },
                )
                for symbol in self.crypto_symbols
            }

            updated_points = [str(entry.get("updated_at")) for entry in symbol_entries.values() if entry.get("updated_at")]
            self.sentiment_scores = {
                "last_updated": max(updated_points) if updated_points else None,
                "configured_sources": configured_sources,
                "newsapi": news_by_symbol,
                "cryptopanic": cryptopanic_by_symbol,
                "reddit": reddit_by_symbol,
                "x": x_by_symbol,
                "symbols": symbol_entries,
            }

            self._persist_scores()
            await asyncio.sleep(loop_interval)

    async def run_event_risk_loop(self, interval: int | None = None):
        loop_interval = max(60, int(interval if interval is not None else self.config.get("event_risk_loop_interval_seconds", 300)))
        while self._running:
            logging.info("Running event risk fetch loop...")
            macro_events, macro_providers = await self._collect_macro_events()
            crypto_events, crypto_providers = await self._collect_crypto_calendar_events()
            configured_providers = macro_providers + crypto_providers
            snapshot = self.event_risk_service.build_snapshot(
                provider_events=macro_events + crypto_events,
                symbols=self.crypto_symbols,
                configured_providers=configured_providers,
            )
            self._persist_event_risk(snapshot)
            for symbol in self.crypto_symbols:
                get_state().update_feed("crypto", "event_risk", symbol)
            logging.info(
                "Event risk updated for %s using %s",
                ", ".join(self.crypto_symbols) or "none",
                ", ".join(configured_providers) or "no_providers",
            )
            await asyncio.sleep(loop_interval)

    async def run_dummy_task(self, label: str = "heartbeat", interval: int = 10):
        """
        Simple repeating log task for dev/testing.
        """
        while self._running:
            logging.info(f"[{label}] heartbeat alive")
            await asyncio.sleep(interval)

    def stop(self):
        self._running = False

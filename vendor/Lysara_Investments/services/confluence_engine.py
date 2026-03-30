from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean, pstdev
from typing import Any, Awaitable, Callable

from api.binance_candles import fetch_binance_public_candles


TIMEFRAME_ORDER = ("1m", "5m", "15m", "1h", "4h")
_CandleFetcher = Callable[[str, str, int], Awaitable[list[dict[str, Any]]]]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous


def _trend_label(score: float) -> str:
    if score >= 0.25:
        return "bullish"
    if score <= -0.25:
        return "bearish"
    return "neutral"


@dataclass
class _CacheEntry:
    expires_at: float
    candles: list[dict[str, Any]]


class ConfluenceEngine:
    def __init__(
        self,
        *,
        default_symbols: list[str] | None = None,
        fetch_candles: _CandleFetcher = fetch_binance_public_candles,
        cache_ttl_seconds: int = 20,
        candle_limit: int = 120,
    ):
        self.default_symbols = [str(symbol).upper() for symbol in (default_symbols or ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"])]
        self.fetch_candles = fetch_candles
        self.cache_ttl_seconds = max(5, int(cache_ttl_seconds))
        self.candle_limit = max(40, int(candle_limit))
        self._cache: dict[tuple[str, str], _CacheEntry] = {}

    async def _get_candles(self, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        key = (str(symbol).upper(), timeframe)
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached and cached.expires_at > now:
            return cached.candles
        candles = await self.fetch_candles(symbol, timeframe, self.candle_limit)
        self._cache[key] = _CacheEntry(expires_at=now + self.cache_ttl_seconds, candles=candles)
        return candles

    def _timeframe_signal(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        closes = [float(candle.get("close") or 0.0) for candle in candles if float(candle.get("close") or 0.0) > 0]
        highs = [float(candle.get("high") or 0.0) for candle in candles if float(candle.get("high") or 0.0) > 0]
        lows = [float(candle.get("low") or 0.0) for candle in candles if float(candle.get("low") or 0.0) > 0]
        if len(closes) < 8:
            return {
                "trend": "unavailable",
                "trend_score": 0.0,
                "change_pct": 0.0,
                "volatility_pct": 0.0,
                "close": closes[-1] if closes else 0.0,
                "support": min(lows) if lows else 0.0,
                "resistance": max(highs) if highs else 0.0,
                "sample_size": len(closes),
            }

        short_window = min(8, max(3, len(closes) // 6))
        long_window = min(24, max(short_window + 3, len(closes) // 3))
        short_ma = fmean(closes[-short_window:])
        long_ma = fmean(closes[-long_window:])
        change_pct = _safe_pct_change(closes[-1], closes[max(0, len(closes) - 5)])
        full_change_pct = _safe_pct_change(closes[-1], closes[0])
        returns = [_safe_pct_change(closes[idx], closes[idx - 1]) for idx in range(1, len(closes))]
        volatility_pct = pstdev(returns) if len(returns) > 1 else 0.0
        ma_gap_pct = _safe_pct_change(short_ma, long_ma) if long_ma else 0.0
        score_raw = (ma_gap_pct * 18.0) + (change_pct * 6.0) + (full_change_pct * 2.0)
        trend_score = _clamp(score_raw, -1.0, 1.0)

        return {
            "trend": _trend_label(trend_score),
            "trend_score": round(trend_score, 3),
            "change_pct": round(change_pct, 4),
            "volatility_pct": round(volatility_pct, 4),
            "close": closes[-1],
            "support": min(lows[-20:]) if lows else 0.0,
            "resistance": max(highs[-20:]) if highs else 0.0,
            "sample_size": len(closes),
        }

    def _nearest_level(self, *, price: float, levels: list[float], side: str) -> float:
        if not levels:
            return price
        if side == "support":
            lower = [level for level in levels if level <= price]
            return max(lower) if lower else min(levels)
        upper = [level for level in levels if level >= price]
        return min(upper) if upper else max(levels)

    def _alignment_label(self, score: float, agreement: float) -> str:
        if score >= 0.35 and agreement >= 0.6:
            return "bullish"
        if score <= -0.35 and agreement >= 0.6:
            return "bearish"
        if abs(score) < 0.15:
            return "neutral"
        return "mixed"

    async def _analyze_symbol(self, symbol: str) -> dict[str, Any]:
        symbol_key = str(symbol or "").upper()
        candles_by_tf = {
            timeframe: candles
            for timeframe, candles in zip(
                TIMEFRAME_ORDER,
                await asyncio.gather(*[self._get_candles(symbol_key, timeframe) for timeframe in TIMEFRAME_ORDER]),
            )
        }

        timeframe_signals = {
            timeframe: self._timeframe_signal(candles)
            for timeframe, candles in candles_by_tf.items()
        }

        valid_scores = [float(signal.get("trend_score") or 0.0) for signal in timeframe_signals.values() if signal.get("trend") != "unavailable"]
        if not valid_scores:
            return {
                "symbol": symbol_key,
                "price": 0.0,
                "alignment_label": "unavailable",
                "confluence_score": 0.0,
                "confidence": 0.0,
                "breakout_probability": 0.0,
                "mean_reversion_probability": 0.0,
                "support": 0.0,
                "resistance": 0.0,
                "timeframes": timeframe_signals,
                "updated_at": _utcnow_iso(),
            }

        price = next(
            (
                float(timeframe_signals[timeframe].get("close") or 0.0)
                for timeframe in TIMEFRAME_ORDER
                if float(timeframe_signals[timeframe].get("close") or 0.0) > 0
            ),
            0.0,
        )

        avg_score = fmean(valid_scores)
        dominant_sign = 1 if avg_score > 0.1 else -1 if avg_score < -0.1 else 0
        directional_scores = [score for score in valid_scores if abs(score) >= 0.15]
        if dominant_sign == 0:
            agreement = len([score for score in valid_scores if abs(score) < 0.25]) / max(len(valid_scores), 1)
        else:
            agreement = len([score for score in directional_scores if math.copysign(1, score) == dominant_sign]) / max(len(directional_scores), 1)

        support_levels: list[float] = []
        resistance_levels: list[float] = []
        for timeframe in ("15m", "1h", "4h"):
            signal = timeframe_signals.get(timeframe) or {}
            support = float(signal.get("support") or 0.0)
            resistance = float(signal.get("resistance") or 0.0)
            if support > 0:
                support_levels.append(support)
            if resistance > 0:
                resistance_levels.append(resistance)

        support = self._nearest_level(price=price, levels=support_levels, side="support")
        resistance = self._nearest_level(price=price, levels=resistance_levels, side="resistance")
        range_pct = abs(resistance - support) / price if price > 0 else 0.0
        distance_to_support = abs(price - support) / price if price > 0 else 0.0
        distance_to_resistance = abs(resistance - price) / price if price > 0 else 0.0
        nearest_level_distance = min(distance_to_support, distance_to_resistance) if price > 0 else 1.0
        proximity_factor = 1.0 - _clamp(nearest_level_distance / max(range_pct, 0.005), 0.0, 1.0)
        average_volatility = fmean(
            [float(signal.get("volatility_pct") or 0.0) for signal in timeframe_signals.values() if signal.get("trend") != "unavailable"]
        )
        compression_factor = 1.0 - _clamp(average_volatility / 0.02, 0.0, 1.0)

        breakout_probability = _clamp(
            0.15 + (0.35 * abs(avg_score)) + (0.2 * agreement) + (0.15 * proximity_factor) + (0.15 * compression_factor),
            0.05,
            0.95,
        )
        mean_reversion_probability = _clamp(
            0.15 + (0.3 * (1.0 - agreement)) + (0.2 * (1.0 - abs(avg_score))) + (0.35 * proximity_factor),
            0.05,
            0.95,
        )
        confidence = _clamp(
            0.2 + (0.35 * agreement) + (0.2 * min(len(valid_scores) / len(TIMEFRAME_ORDER), 1.0)) + (0.25 * (1.0 - min(range_pct / 0.08, 1.0))),
            0.05,
            0.95,
        )

        bullish_count = len([score for score in valid_scores if score >= 0.25])
        bearish_count = len([score for score in valid_scores if score <= -0.25])
        neutral_count = len(valid_scores) - bullish_count - bearish_count

        return {
            "symbol": symbol_key,
            "price": round(price, 6),
            "alignment_label": self._alignment_label(avg_score, agreement),
            "confluence_score": round(avg_score, 3),
            "confidence": round(confidence, 3),
            "breakout_probability": round(breakout_probability, 3),
            "mean_reversion_probability": round(mean_reversion_probability, 3),
            "support": round(support, 6) if support else 0.0,
            "resistance": round(resistance, 6) if resistance else 0.0,
            "range_width_pct": round(range_pct, 4),
            "timeframe_alignment": {
                "bullish": bullish_count,
                "bearish": bearish_count,
                "neutral": neutral_count,
            },
            "timeframes": timeframe_signals,
            "updated_at": _utcnow_iso(),
        }

    async def get_confluence(self, symbols: list[str] | None = None) -> dict[str, Any]:
        requested = [str(symbol).upper() for symbol in (symbols or self.default_symbols) if str(symbol).strip()]
        requested = list(dict.fromkeys(requested))
        items = await asyncio.gather(*[self._analyze_symbol(symbol) for symbol in requested])
        items.sort(key=lambda item: abs(float(item.get("confluence_score") or 0.0)), reverse=True)
        return {
            "updated_at": _utcnow_iso(),
            "timeframes": list(TIMEFRAME_ORDER),
            "symbols": items,
        }

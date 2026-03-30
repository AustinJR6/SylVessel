from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import aiohttp

from api.binance_public import _binance_symbol


BINANCE_PUBLIC_KLINES_URL = "https://api.binance.us/api/v3/klines"
SUPPORTED_INTERVALS = {"1m", "5m", "15m", "1h", "4h"}


async def fetch_binance_public_candles(symbol: str, interval: str, limit: int = 120) -> List[Dict[str, Any]]:
    market_symbol = _binance_symbol(symbol)
    normalized_interval = str(interval or "").strip()
    if not market_symbol or normalized_interval not in SUPPORTED_INTERVALS:
        return []

    params = {
        "symbol": market_symbol,
        "interval": normalized_interval,
        "limit": max(20, min(int(limit or 120), 500)),
    }

    for attempt in range(1, 4):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(BINANCE_PUBLIC_KLINES_URL, params=params) as resp:
                    data = await resp.json(content_type=None)
                    if resp.status >= 400 or not isinstance(data, list):
                        logging.warning(
                            "Binance public candles HTTP %s for %s %s: %s",
                            resp.status,
                            market_symbol,
                            normalized_interval,
                            data,
                        )
                        await asyncio.sleep(attempt)
                        continue

                    candles: List[Dict[str, Any]] = []
                    for row in data:
                        if not isinstance(row, list) or len(row) < 7:
                            continue
                        try:
                            candles.append(
                                {
                                    "open_time": int(row[0]),
                                    "open": float(row[1]),
                                    "high": float(row[2]),
                                    "low": float(row[3]),
                                    "close": float(row[4]),
                                    "volume": float(row[5]),
                                    "close_time": int(row[6]),
                                }
                            )
                        except Exception:
                            continue
                    return candles
        except Exception as exc:
            logging.warning(
                "Binance public candles fetch failed for %s %s (attempt %s): %s",
                market_symbol,
                normalized_interval,
                attempt,
                exc,
            )
            await asyncio.sleep(attempt)

    return []

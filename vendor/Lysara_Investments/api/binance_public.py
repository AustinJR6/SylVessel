from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

import aiohttp


BINANCE_PUBLIC_TICKER_URL = "https://api.binance.us/api/v3/ticker/bookTicker"


def _binance_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("-", "")


async def fetch_binance_public_price(symbol: str) -> Dict[str, Any]:
    market_symbol = _binance_symbol(symbol)
    if not market_symbol:
        return {"symbol": symbol, "price": 0.0, "bid": 0.0, "ask": 0.0, "source": "binance_public"}

    params = {"symbol": market_symbol}
    for attempt in range(1, 4):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(BINANCE_PUBLIC_TICKER_URL, params=params) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        logging.warning("Binance public ticker HTTP %s for %s: %s", resp.status, market_symbol, data)
                        await asyncio.sleep(attempt)
                        continue
                    bid = float(data.get("bidPrice", 0) or 0)
                    ask = float(data.get("askPrice", 0) or 0)
                    price = (bid + ask) / 2 if bid and ask else bid or ask
                    return {
                        "symbol": symbol,
                        "price": float(price or 0.0),
                        "bid": bid,
                        "ask": ask,
                        "source": "binance_public",
                    }
        except Exception as exc:
            logging.warning("Binance public price fetch failed for %s (attempt %s): %s", market_symbol, attempt, exc)
            await asyncio.sleep(attempt)

    return {"symbol": symbol, "price": 0.0, "bid": 0.0, "ask": 0.0, "source": "binance_public"}

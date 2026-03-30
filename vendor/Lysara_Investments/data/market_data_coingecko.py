# data/market_data_coingecko.py
"""Fetch current cryptocurrency prices from the public CoinGecko API."""

import asyncio
import logging
from datetime import datetime

import aiohttp

from .price_cache import get_price, update_price
from services.daemon_state import get_state


async def fetch_coingecko_price(session: aiohttp.ClientSession, coin_id: str) -> dict:
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    try:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            price = data.get(coin_id, {}).get("usd", 0.0)
            if price:
                update_price(f"{coin_id.upper()}-USD", price, "coingecko")
                get_state().update_feed("crypto", "coingecko", f"{coin_id.upper()}-USD")
            else:
                logging.debug(f"CoinGecko returned zero price for {coin_id}")
            return {
                "symbol": coin_id,
                "price": price,
                "time": datetime.utcnow().isoformat(),
            }
    except Exception as e:
        logging.error(f"Coingecko price fetch failed: {e}")
        return {"symbol": coin_id, "price": 0.0, "time": datetime.utcnow().isoformat()}


async def start_coingecko_polling(symbols: list[str], interval: int = 60, on_data=None):
    """Poll CoinGecko every ``interval`` seconds for crypto symbols.

    Stock tickers are ignored. Prices already provided by Binance take
    precedence over CoinGecko data.
    """
    logging.info("CoinGecko price polling disabled")
    while True:
        await asyncio.sleep(interval)


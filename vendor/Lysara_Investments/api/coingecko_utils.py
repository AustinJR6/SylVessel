"""Utility for fetching prices from CoinGecko."""

import aiohttp
import asyncio
import logging

FALLBACK_IDS = {
    "ADA": "cardano",
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

async def get_price(symbol: str) -> float:
    """Return the USD price for a ticker symbol via CoinGecko."""
    ticker = symbol.split("-")[0].upper()
    coin_id = FALLBACK_IDS.get(ticker, ticker.lower())
    params = {"ids": coin_id, "vs_currencies": "usd"}
    for attempt in range(1, 4):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(COINGECKO_URL, params=params) as resp:
                    data = await resp.json()
                    price = data.get(coin_id, {}).get("usd")
                    if price is not None:
                        return float(price)
        except Exception as e:
            logging.error(
                f"CoinGecko price fetch failed (attempt {attempt}): {e}"
            )
            await asyncio.sleep(attempt)
    return 0.0

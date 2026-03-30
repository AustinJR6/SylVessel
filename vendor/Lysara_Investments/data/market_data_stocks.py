# data/market_data_stocks.py

import asyncio
import logging
from datetime import datetime

from .price_cache import update_price
from services.alpaca_manager import AlpacaManager
from services.daemon_state import get_state


async def fetch_stock_prices(alpaca: AlpacaManager, symbols: list[str]):
    """Fetch latest stock prices using Alpaca market data."""
    results = []
    for symbol in symbols:
        try:
            logging.debug(f"Requesting price for {symbol} from Alpaca")
            price = (await alpaca.fetch_market_price(symbol)).get("price", 0)
            update_price(symbol, price, "alpaca")
            get_state().update_feed("stocks", "alpaca_poll", symbol)
            results.append({
                "symbol": symbol,
                "price": price,
                "time": datetime.utcnow().isoformat(),
            })
        except Exception as e:
            logging.error(f"Failed to fetch price for {symbol}: {e}")
    return results


async def start_stock_polling_loop(symbols, alpaca: AlpacaManager, interval=10, on_price=None):
    """Poll stock prices every `interval` seconds via Alpaca."""
    logging.info(
        f"Starting Alpaca polling loop for: {', '.join(symbols)} every {interval}s"
    )
    while True:
        prices = await fetch_stock_prices(alpaca, symbols)
        for p in prices:
            if on_price:
                await on_price(p)
        await asyncio.sleep(interval)

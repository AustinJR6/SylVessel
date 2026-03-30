import asyncio
import logging
from datetime import datetime

from api.binance_public import fetch_binance_public_price
from .price_cache import update_price
from services.daemon_state import get_state


def _is_crypto_symbol(symbol: str) -> bool:
    return "-" in str(symbol or "").strip().upper()


async def handle_market_message(message: dict):
    logging.info(f"[CRYPTO WS] Ticker update: {message.get('symbol')} @ {message.get('price')}")


async def start_crypto_market_feed(symbols: list[str], crypto_api, on_message=handle_market_message, poll_interval: int = 5):
    """Poll Binance public market data and refresh the shared price cache."""
    crypto_symbols = [str(symbol).strip().upper() for symbol in symbols if _is_crypto_symbol(symbol)]

    while True:
        try:
            for canonical in crypto_symbols:
                data = await fetch_binance_public_price(canonical)
                price = float(data.get("price", 0.0) or 0.0)
                if not price:
                    if crypto_api is not None:
                        data = await crypto_api.fetch_market_price(canonical)
                        price = float(data.get("price", 0.0) or 0.0)
                    if not price:
                        logging.debug("Crypto market feed received zero price for %s", canonical)
                        continue
                source = str(data.get("source") or "binance_public")
                update_price(canonical, price, source)
                get_state().update_feed("crypto", f"{source}_poll", canonical)
                await on_message({
                    "symbol": canonical,
                    "price": price,
                    "source": source,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            await asyncio.sleep(poll_interval)
        except Exception as e:
            logging.error(f"Crypto market polling error: {e}")
            logging.info("Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

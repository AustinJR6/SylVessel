# data/market_data_forex.py

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from .price_cache import update_price
from services.daemon_state import get_state

async def fetch_forex_prices(session, instruments: list[str], api_key: str, account_id: str):
    """
    Pulls the latest prices from OANDA REST endpoint. Replace with WebSocket later if needed.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        url = f"https://api-fxpractice.oanda.com/v3/accounts/{account_id}/pricing?instruments={','.join(instruments)}"
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get("prices", [])
    except Exception as e:
        logging.error(f"Forex REST price fetch failed: {e}")
        return []

async def start_forex_polling_loop(instruments, api_key, account_id, interval=5, on_price=None):
    """
    Polls OANDA for forex prices every `interval` seconds.
    """
    async with aiohttp.ClientSession() as session:
        while True:
            prices = await fetch_forex_prices(session, instruments, api_key, account_id)
            now = datetime.utcnow().isoformat()
            for p in prices:
                instrument = p.get("instrument")
                bid = p.get("bids", [{}])[0].get("price")
                ask = p.get("asks", [{}])[0].get("price")
                try:
                    mid = (float(bid) + float(ask)) / 2.0
                    if instrument:
                        update_price(instrument, mid, "oanda")
                        get_state().update_feed("forex", "oanda_poll", instrument)
                except (TypeError, ValueError):
                    mid = None
                if on_price:
                    await on_price({
                        "instrument": instrument,
                        "bid": bid,
                        "ask": ask,
                        "price": mid,
                        "time": now
                    })
            await asyncio.sleep(interval)

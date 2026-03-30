# data/market_data_alpaca.py
"""Real-time stock market data streaming via Alpaca's websocket API."""

import asyncio
import json
import logging

import websockets
from .price_cache import update_price
from services.daemon_state import get_state


async def start_stock_ws_feed(
    symbols: list[str],
    api_key: str,
    api_secret: str,
    base_url: str,
    data_feed: str = "iex",
    on_bar=None,
):
    """Run a websocket loop streaming live bars from Alpaca."""

    async def handle_bar(bar):
        data = {
            "symbol": bar["symbol"],
            "price": float(bar["close"]),
            "time": bar.get("timestamp", ""),
        }
        update_price(data["symbol"], data["price"], "alpaca_ws")
        get_state().update_feed("stocks", "alpaca_ws", data["symbol"])
        if on_bar:
            await on_bar(data)
        else:
            logging.info(f"[ALPACA WS] {data['symbol']} @ {data['price']}")

    async def handle_event(event):
        event_type = event.get("T")
        if event_type in {"success", "subscription"}:
            logging.info(f"[ALPACA WS] {event_type}: {event}")
            return
        if event_type == "error":
            logging.error(f"Alpaca WS error event: {event}")
            return
        if event_type == "b":
            parsed = {
                "symbol": event.get("S"),
                "close": event.get("c"),
                "timestamp": event.get("t"),
            }
            await handle_bar(parsed)

    url = f"wss://stream.data.alpaca.markets/v2/{data_feed}"
    logging.info(
        f"Connecting to Alpaca WS feed {url} for symbols: {', '.join(symbols)}"
    )

    while True:
        try:
            async with websockets.connect(url) as ws:
                auth = {"action": "auth", "key": api_key, "secret": api_secret}
                await ws.send(json.dumps(auth))
                subs = {"action": "subscribe", "bars": symbols}
                await ws.send(json.dumps(subs))

                async for msg in ws:
                    logging.debug(f"Alpaca WS raw: {msg}")
                    data = json.loads(msg)
                    if isinstance(data, list):
                        for event in data:
                            if isinstance(event, dict):
                                await handle_event(event)
                    elif isinstance(data, dict):
                        if data.get("bars"):
                            for bar in data.get("bars", []):
                                if isinstance(bar, dict):
                                    await handle_bar(
                                        {
                                            "symbol": bar.get("S"),
                                            "close": bar.get("c"),
                                            "timestamp": bar.get("t"),
                                        }
                                    )
                        else:
                            await handle_event(data)
        except Exception as e:
            logging.error(f"Alpaca WS error: {e}")
            await asyncio.sleep(5)


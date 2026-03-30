import os
import time
import json
import logging
import asyncio
from typing import Dict, Any

import requests
from urllib.parse import urljoin
from dotenv import load_dotenv

load_dotenv()


def _normalize_base_url(url: str) -> str:
    """Normalize Alpaca base URL so endpoint paths can safely include /v2."""
    base = (url or "").rstrip("/")
    if base.endswith("/v2"):
        base = base[:-3]
    return base


def _data_base_url() -> str:
    base = (os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets") or "").rstrip("/")
    if base.endswith("/v2"):
        base = base[:-3]
    return base


def _headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
        "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
    }


def _request(method: str, path: str, *, live: bool = False, **kwargs) -> Dict[str, Any]:
    base_url = _normalize_base_url(
        os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
        if live
        else os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    )
    url = urljoin(base_url.rstrip('/') + '/', path.lstrip('/'))
    for attempt in range(1, 4):
        try:
            resp = requests.request(method, url, headers=_headers(), timeout=10, **kwargs)
            resp.raise_for_status()
            if resp.text:
                return resp.json()
            return {}
        except Exception as e:
            logging.error(f"{method} {url} failed (attempt {attempt}): {e}")
            time.sleep(2 ** (attempt - 1))
    return {}


async def get_account(live: bool = False) -> Dict[str, Any]:
    return await asyncio.to_thread(_request, "GET", "/v2/account", live=live)


async def get_positions(live: bool = False) -> Any:
    return await asyncio.to_thread(_request, "GET", "/v2/positions", live=live)


async def place_order(
    symbol: str,
    side: str,
    qty: float,
    type: str = "market",
    time_in_force: str = "gtc",
    live: bool = False,
) -> Dict[str, Any]:
    body = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "type": type,
        "time_in_force": time_in_force,
    }
    return await asyncio.to_thread(
        _request, "POST", "/v2/orders", json=body, live=live
    )


async def cancel_order(order_id: str, live: bool = False) -> Dict[str, Any]:
    path = f"/v2/orders/{order_id}"
    return await asyncio.to_thread(_request, "DELETE", path, live=live)


async def fetch_market_price(symbol: str, live: bool = False) -> Dict[str, Any]:
    path = f"/v2/stocks/{symbol}/trades/latest"
    url = urljoin(_data_base_url().rstrip("/") + "/", path.lstrip("/"))
    data = {}
    for attempt in range(1, 4):
        try:
            data = await asyncio.to_thread(requests.get, url, headers=_headers(), timeout=10)
            data.raise_for_status()
            data = data.json() if data.text else {}
            break
        except Exception as e:
            logging.error(f"GET {url} failed (attempt {attempt}): {e}")
            time.sleep(2 ** (attempt - 1))
            data = {}
    price = 0.0
    if isinstance(data, dict):
        price = float(data.get("trade", {}).get("p", 0))
    return {"price": price}

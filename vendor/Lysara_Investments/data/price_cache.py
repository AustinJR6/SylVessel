from __future__ import annotations

"""In-memory cache of the latest ticker prices by symbol."""

from collections import deque
from datetime import datetime
from threading import Lock
from typing import Dict, Any, List

# key -> {'price': float, 'source': str, 'time': ISO8601}
_PRICE_CACHE: Dict[str, Dict[str, str | float]] = {}
_PRICE_HISTORY: Dict[str, deque] = {}
_LOCK = Lock()
_MAX_POINTS = 10_000


def update_price(symbol: str, price: float, source: str) -> None:
    """Update cached price for a symbol.

    Parameters
    ----------
    symbol : str
        Canonical symbol like ``BTC-USD`` or ``AAPL``.
    price : float
        Latest trade/last price.
    source : str
        Data source identifier such as ``binance`` or ``alpaca``.
    """
    ts = datetime.utcnow().isoformat()
    sym = symbol.upper()
    with _LOCK:
        _PRICE_CACHE[sym] = {
            "price": float(price),
            "source": source,
            "time": ts,
        }
        if sym not in _PRICE_HISTORY:
            _PRICE_HISTORY[sym] = deque(maxlen=_MAX_POINTS)
        _PRICE_HISTORY[sym].append(
            {
                "time": ts,
                "price": float(price),
                "source": source,
            }
        )


def get_price(symbol: str) -> Dict[str, str | float] | None:
    """Return cached price entry for ``symbol`` if present."""
    with _LOCK:
        return _PRICE_CACHE.get(symbol.upper())


def get_all() -> Dict[str, Dict[str, str | float]]:
    """Return the full cache."""
    with _LOCK:
        return dict(_PRICE_CACHE)


def get_price_series(symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Return recent historical points for a symbol."""
    with _LOCK:
        points = list(_PRICE_HISTORY.get(symbol.upper(), []))
    if limit > 0:
        return points[-limit:]
    return points

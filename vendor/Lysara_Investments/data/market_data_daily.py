from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

from utils.runtime_paths import get_runtime_dir


def _history_path(symbol: str) -> Path:
    safe = str(symbol or "").strip().upper().replace("/", "_")
    return get_runtime_dir() / "daily_history" / f"{safe}.json"


def fetch_daily_history(symbol: str, range_label: str = "6mo", interval: str = "1d") -> list[dict[str, Any]]:
    clean_symbol = str(symbol or "").strip().upper()
    if not clean_symbol:
        return []
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(clean_symbol)}?range={quote(range_label)}&interval={quote(interval)}"
    try:
        with urlopen(url, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        logging.warning("Daily history fetch failed for %s: %s", clean_symbol, exc)
        return []
    result = (((payload or {}).get("chart") or {}).get("result") or [None])[0] if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        return []
    timestamps = list(result.get("timestamp") or [])
    quote_rows = (((result.get("indicators") or {}).get("quote") or [None])[0] or {})
    opens = list(quote_rows.get("open") or [])
    highs = list(quote_rows.get("high") or [])
    lows = list(quote_rows.get("low") or [])
    closes = list(quote_rows.get("close") or [])
    volumes = list(quote_rows.get("volume") or [])
    rows: list[dict[str, Any]] = []
    for index, ts in enumerate(timestamps):
        close = _safe_float(closes[index] if index < len(closes) else None)
        if close <= 0:
            continue
        rows.append(
            {
                "timestamp": int(ts),
                "open": _safe_float(opens[index] if index < len(opens) else close, close),
                "high": _safe_float(highs[index] if index < len(highs) else close, close),
                "low": _safe_float(lows[index] if index < len(lows) else close, close),
                "close": close,
                "volume": _safe_float(volumes[index] if index < len(volumes) else 0.0, 0.0),
            }
        )
    return rows


def save_daily_history(symbol: str, rows: list[dict[str, Any]]) -> None:
    path = _history_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": str(symbol or "").strip().upper(),
        "updated_at": _utcnow_iso(),
        "rows": rows,
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temp_path.replace(path)


def get_daily_history(symbol: str, limit: int = 180) -> list[dict[str, Any]]:
    path = _history_path(symbol)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = list((payload or {}).get("rows") or [])
    if limit > 0:
        return rows[-limit:]
    return rows


async def refresh_symbol_history(symbol: str, range_label: str = "6mo") -> list[dict[str, Any]]:
    rows = await asyncio.to_thread(fetch_daily_history, symbol, range_label)
    if rows:
        await asyncio.to_thread(save_daily_history, symbol, rows)
    return rows


async def run_daily_feed(symbols: list[str], interval_seconds: int = 3600, range_label: str = "6mo") -> None:
    while True:
        try:
            for symbol in [str(item or "").strip().upper() for item in symbols if str(item or "").strip()]:
                await refresh_symbol_history(symbol, range_label=range_label)
        except Exception as exc:
            logging.warning("Daily feed refresh failed: %s", exc)
        await asyncio.sleep(max(300, int(interval_seconds)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _utcnow_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().isoformat()

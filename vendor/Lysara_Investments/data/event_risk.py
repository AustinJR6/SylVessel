from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 20) -> Any:
    req = UrlRequest(url=url, headers=headers or {"Accept": "application/json"}, method="GET")
    with urlopen(req, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload or "{}")


def _parse_datetime(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    variants = [raw]
    if raw.endswith("Z"):
        variants.append(raw.replace("Z", "+00:00"))
    if " " in raw and "T" not in raw:
        variants.append(raw.replace(" ", "T"))
    if len(raw) == 10 and raw.count("-") == 2:
        variants.append(f"{raw}T00:00:00+00:00")

    for candidate in variants:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except Exception:
            continue
    return None


def _impact_score(raw_value: Any) -> tuple[str, float]:
    normalized = str(raw_value or "").strip().lower()
    if normalized in {"3", "high", "3.0", "important", "critical"}:
        return "high", 0.9
    if normalized in {"2", "medium", "2.0", "moderate"}:
        return "medium", 0.65
    if normalized in {"1", "low", "1.0", "minor"}:
        return "low", 0.35
    return "medium", 0.55


async def fetch_finnhub_economic_calendar(api_key: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    if not api_key:
        return []
    params = urlencode({"from": start_date, "to": end_date, "token": api_key})
    url = f"https://finnhub.io/api/v1/calendar/economic?{params}"
    try:
        payload = await asyncio.to_thread(_http_json, url)
    except Exception as exc:
        logging.error("Finnhub economic calendar fetch failed: %s", exc)
        return []

    rows = []
    if isinstance(payload, dict):
        rows = payload.get("economicCalendar") or payload.get("data") or payload.get("events") or []
    elif isinstance(payload, list):
        rows = payload

    events: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("event") or row.get("indicator") or row.get("title") or "").strip()
        starts_at = _parse_datetime(row.get("date") or row.get("time") or row.get("dateTime"))
        if not title or not starts_at:
            continue
        severity, impact_score = _impact_score(row.get("impact") or row.get("importance"))
        country = str(row.get("country") or row.get("region") or "global").strip()
        currency = str(row.get("currency") or "USD").strip().upper()
        events.append(
            {
                "id": f"finnhub:{title}:{starts_at}",
                "title": title,
                "category": "macro",
                "source": "finnhub",
                "starts_at": starts_at,
                "severity": severity,
                "impact_score": impact_score,
                "scope": "macro",
                "symbols": [],
                "tags": [country, currency],
            }
        )
    return events


async def fetch_tradingeconomics_calendar(api_key: str, api_secret: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    if not api_key or not api_secret:
        return []
    params = urlencode({"c": f"{api_key}:{api_secret}", "f": "json", "d1": start_date, "d2": end_date})
    url = f"https://api.tradingeconomics.com/calendar?{params}"
    try:
        payload = await asyncio.to_thread(_http_json, url)
    except Exception as exc:
        logging.error("TradingEconomics calendar fetch failed: %s", exc)
        return []

    rows = payload if isinstance(payload, list) else payload.get("events") if isinstance(payload, dict) else []
    events: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("Event") or row.get("event") or row.get("Title") or "").strip()
        starts_at = _parse_datetime(row.get("Date") or row.get("date") or row.get("CalendarDate"))
        if not title or not starts_at:
            continue
        severity, impact_score = _impact_score(row.get("Importance") or row.get("importance"))
        country = str(row.get("Country") or row.get("country") or "global").strip()
        events.append(
            {
                "id": f"tradingeconomics:{title}:{starts_at}",
                "title": title,
                "category": "macro",
                "source": "tradingeconomics",
                "starts_at": starts_at,
                "severity": severity,
                "impact_score": impact_score,
                "scope": "macro",
                "symbols": [],
                "tags": [country],
            }
        )
    return events


async def fetch_coinmarketcal_events(api_key: str, assets: list[str], max_results: int = 25) -> list[dict[str, Any]]:
    if not api_key or not assets:
        return []
    params = urlencode({"max": max(5, min(int(max_results or 25), 100)), "coins": ",".join(sorted(set(assets)))})
    url = f"https://developers.coinmarketcal.com/v1/events?{params}"
    headers = {"Accept": "application/json", "x-api-key": api_key}
    try:
        payload = await asyncio.to_thread(_http_json, url, headers)
    except Exception as exc:
        logging.error("CoinMarketCal event fetch failed: %s", exc)
        return []

    rows = payload if isinstance(payload, list) else payload.get("body") if isinstance(payload, dict) else []
    if isinstance(rows, dict):
        rows = rows.get("events") or []
    events: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or row.get("name") or "").strip()
        starts_at = _parse_datetime(row.get("date_event") or row.get("created_date") or row.get("startDate"))
        coins = row.get("coins") or row.get("currencies") or []
        symbols = [f"{str(item.get('symbol') or item).strip().upper()}-USD" for item in coins if str(item.get("symbol") if isinstance(item, dict) else item).strip()]
        if not title or not starts_at:
            continue
        severity, impact_score = _impact_score(row.get("hotness") or row.get("importance") or "medium")
        category = str((row.get("category") or {}).get("name") if isinstance(row.get("category"), dict) else row.get("category") or "crypto_event").strip()
        events.append(
            {
                "id": f"coinmarketcal:{row.get('id') or title}:{starts_at}",
                "title": title,
                "category": category,
                "source": "coinmarketcal",
                "starts_at": starts_at,
                "severity": severity,
                "impact_score": impact_score,
                "scope": "crypto",
                "symbols": sorted(dict.fromkeys(symbols)),
                "tags": [category],
            }
        )
    return events

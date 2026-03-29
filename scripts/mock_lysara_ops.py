#!/usr/bin/env python
from __future__ import annotations

import json
import os
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


STATE_LOCK = threading.RLock()
DEFAULT_CRYPTO_WATCHLIST = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"]
DEFAULT_STOCK_WATCHLIST = ["AAPL", "MSFT", "GOOG", "AMZN"]
CRYPTO_SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "ADA-USD": "ADAUSDT",
}
DEFAULT_STRATEGY_REGISTRY = {
    "stocks": [
        {
            "strategy_name": "StockMomentumStrategy",
            "symbols": ["AAPL", "MSFT"],
            "enabled": True,
            "params": {"type": "stock_momentum", "trade_symbols": ["AAPL", "MSFT"], "lookback": 20},
        }
    ],
    "crypto": [
        {
            "strategy_name": "MomentumStrategy",
            "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "enabled": True,
            "params": {"type": "momentum", "trade_symbols": ["BTC-USD", "ETH-USD", "SOL-USD"], "lookback": 24},
        }
    ],
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _future(hours: float) -> str:
    return (_utc_now() + timedelta(hours=hours)).isoformat()


def _parse_iso(value: Optional[str]) -> datetime:
    text = str(value or "").strip()
    if not text:
        return _utc_now()
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return _utc_now()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_starting_balance() -> float:
    return max(10.0, _safe_float(os.getenv("MOCK_LYSARA_STARTING_BALANCE"), 1000.0))


def _state_path() -> Optional[Path]:
    raw = str(os.getenv("MOCK_LYSARA_STATE_PATH") or "").strip()
    return Path(raw) if raw else None


def _default_state(starting_balance: Optional[float] = None) -> Dict[str, Any]:
    now = _utc_now()
    now_iso = now.isoformat()
    starting = max(10.0, float(starting_balance or _env_starting_balance()))
    crypto_equity = round(starting * 0.5, 2)
    stocks_equity = round(starting - crypto_equity, 2)
    return {
        "started_at": now_iso,
        "status": {
            "mock_mode": True,
            "paused": False,
            "pause_reason": "",
            "simulation_mode": True,
            "live_trading_enabled": False,
            "autonomous_mode": True,
            "active_markets": ["stocks", "crypto"],
            "regime": "neutral",
            "equity": {"stocks": stocks_equity, "crypto": crypto_equity},
            "timestamp": now_iso,
            "updated_at": now_iso,
            "started_at": now_iso,
            "last_heartbeat_at": now_iso,
            "uptime_seconds": 0,
            "last_heartbeat_ago": 0.0,
            "feed_freshness": {
                "stocks": {"alpaca_poll:AAPL": 2.0, "alpaca_poll:MSFT": 2.5},
                "crypto": {"binance_public_poll:BTC-USD": 1.5, "binance_public_poll:ETH-USD": 1.8, "binance_public_poll:SOL-USD": 2.1},
            },
            "broker_health": {
                "stocks": {"connected": True, "detail": "mock_account_ok", "updated_at": now_iso},
                "crypto": {"connected": True, "detail": "mock_account_ok", "updated_at": now_iso},
            },
            "strategy_registry": deepcopy(DEFAULT_STRATEGY_REGISTRY),
            "symbol_controls": {
                "stocks": {symbol: True for symbol in DEFAULT_STOCK_WATCHLIST},
                "crypto": {symbol: True for symbol in DEFAULT_CRYPTO_WATCHLIST},
            },
            "strategy_controls": {
                "stocks": {"StockMomentumStrategy": True},
                "crypto": {"MomentumStrategy": True},
            },
            "risk_managers": {"stocks": 1, "crypto": 1},
            "simulation_portfolio": {
                "currency": "USD",
                "starting_balance": starting,
                "balance": starting,
                "cash": starting,
                "buying_power": starting,
                "portfolio_value": starting,
            },
        },
        "portfolio": {
            "simulation_mode": True,
            "mock_mode": True,
            "total_equity": starting,
            "simulation_portfolio": {
                "currency": "USD",
                "starting_balance": starting,
                "balance": starting,
                "cash": starting,
                "buying_power": starting,
                "portfolio_value": starting,
            },
            "markets": {
                "stocks": {"positions": []},
                "crypto": {"positions": []},
            },
        },
        "watchlists": {
            "stocks": list(DEFAULT_STOCK_WATCHLIST),
            "crypto": list(DEFAULT_CRYPTO_WATCHLIST),
        },
        "strategy_candidates": [],
        "positions": {"items": []},
        "incidents": {"items": []},
        "market_snapshot": {
            "market": "mixed",
            "prices": {
                "AAPL": {"price": 192.5, "change_pct_24h": 1.2, "trend_score": 1.0, "source": "bootstrap_cache", "timestamp": now_iso},
                "MSFT": {"price": 418.2, "change_pct_24h": 0.8, "trend_score": 0.4, "source": "bootstrap_cache", "timestamp": now_iso},
                "BTC-USD": {"price": 69000.0, "change_pct_24h": 1.6, "source": "bootstrap_cache", "timestamp": now_iso},
                "ETH-USD": {"price": 3520.0, "change_pct_24h": 1.1, "source": "bootstrap_cache", "timestamp": now_iso},
                "SOL-USD": {"price": 171.2, "change_pct_24h": 2.1, "source": "bootstrap_cache", "timestamp": now_iso},
            },
            "feed_freshness": {
                "stocks": {"alpaca_poll:AAPL": 2.0, "alpaca_poll:MSFT": 2.5},
                "crypto": {"binance_public_poll:BTC-USD": 1.5, "binance_public_poll:ETH-USD": 1.8, "binance_public_poll:SOL-USD": 2.1},
            },
            "updated_at": now_iso,
            "feed_sources": {},
        },
        "sentiment": {
            "updated_at": now_iso,
            "configured_sources": ["mock_news", "mock_social"],
            "symbols": [
                {
                    "symbol": "BTC-USD",
                    "display_name": "Bitcoin",
                    "asset": "BTC",
                    "score": 0.18,
                    "confidence": 0.67,
                    "mention_velocity": 32,
                    "source_count": 3,
                    "source_coverage": 0.75,
                    "anomaly_flags": [],
                    "sources": {"mock_news": {"score": 0.15, "count": 12}},
                    "updated_at": now_iso,
                }
            ],
        },
        "confluence": {
            "updated_at": now_iso,
            "timeframes": ["1m", "5m", "15m", "1h", "4h"],
            "symbols": [
                {
                    "symbol": "BTC-USD",
                    "alignment_label": "bullish",
                    "confluence_score": 0.61,
                    "confidence": 0.72,
                    "breakout_probability": 0.56,
                    "mean_reversion_probability": 0.22,
                    "support": 68000.0,
                    "resistance": 69500.0,
                    "range_width_pct": 0.022,
                    "timeframe_alignment": {"bullish": 4, "bearish": 0, "neutral": 1},
                    "timeframes": {
                        "1m": {"trend": "bullish", "trend_score": 0.28},
                        "5m": {"trend": "bullish", "trend_score": 0.41},
                    },
                }
            ],
        },
        "event_risk": {
            "updated_at": now_iso,
            "configured_providers": ["mock_events"],
            "lookahead_hours": 24,
            "warning_threshold": 0.45,
            "block_threshold": 0.7,
            "reduction_threshold": 0.85,
            "reduction_factor": 0.5,
            "events": [
                {
                    "id": "event-1",
                    "title": "Bitcoin ETF Vote",
                    "category": "crypto_event",
                    "source": "mock_events",
                    "starts_at": _future(8),
                    "severity": "high",
                    "impact_score": 0.58,
                    "scope": "crypto",
                    "symbols": ["BTC-USD"],
                    "tags": ["ETF"],
                    "hours_until": 8.0,
                }
            ],
            "symbols": [
                {
                    "symbol": "BTC-USD",
                    "risk_score": 0.58,
                    "action": "warn",
                    "block_new_positions": False,
                    "reduce_position_pct": 0.0,
                    "primary_event_key": "event-1",
                    "upcoming_events": [{"id": "event-1", "title": "Bitcoin ETF Vote", "starts_at": _future(8)}],
                    "reasons": ["Bitcoin ETF Vote @ soon"],
                }
            ],
        },
        "override": {
            "enabled": False,
            "actor": "",
            "reason": "",
            "activated_at": None,
            "expires_at": None,
            "ttl_seconds": 0,
            "allowed_controls": [],
            "last_cleared_at": None,
        },
        "recent_trades": {"items": [], "trades": []},
        "trade_lookup": {},
        "submitted_trade_intents": [],
        "journal": [],
        "research": [],
    }


def _merge(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = dict(existing)
        for key, value in incoming.items():
            merged[key] = _merge(existing.get(key), value)
        return merged
    return incoming


def _persist_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    if path and path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            base = _default_state(starting_balance=_safe_float(((loaded.get("status") or {}).get("simulation_portfolio") or {}).get("starting_balance"), _env_starting_balance()))
            state = _merge(base, loaded)
            return state
        except Exception:
            pass
    state = _default_state()
    _persist_state(state)
    return state


STATE: Dict[str, Any] = _load_state()


def _fetch_json(url: str, timeout: float = 8.0) -> Dict[str, Any] | list[Any] | None:
    req = Request(url, headers={"User-Agent": "SylanaVessel/1.0"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def _tracked_symbols(state: Dict[str, Any]) -> Dict[str, list[str]]:
    watchlists = state.get("watchlists") or {}
    tracked = {
        "stocks": list(watchlists.get("stocks") or []),
        "crypto": list(watchlists.get("crypto") or []),
    }
    registry = ((state.get("status") or {}).get("strategy_registry") or {})
    controls = ((state.get("status") or {}).get("strategy_controls") or {})
    for market in ("stocks", "crypto"):
        enabled_map = controls.get(market) or {}
        for item in registry.get(market) or []:
            if not bool(enabled_map.get(str(item.get("strategy_name") or ""), item.get("enabled", True))):
                continue
            for symbol in item.get("symbols") or []:
                clean = str(symbol or "").strip().upper()
                if clean and clean not in tracked[market]:
                    tracked[market].append(clean)
    for row in ((state.get("positions") or {}).get("items") or []):
        symbol = str(row.get("symbol") or "").strip().upper()
        market = str(row.get("market") or ("crypto" if symbol.endswith("-USD") else "stocks")).strip().lower()
        if symbol and market in tracked and symbol not in tracked[market]:
            tracked[market].append(symbol)
    return tracked


def _refresh_stock_quotes(symbols: list[str]) -> Dict[str, Dict[str, Any]]:
    requested = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    if not requested:
        return {}
    payload = _fetch_json(
        "https://query1.finance.yahoo.com/v7/finance/quote?symbols=" + ",".join(requested)
    )
    rows = (((payload or {}).get("quoteResponse") or {}).get("result") or []) if isinstance(payload, dict) else []
    mapped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        price = _safe_float(row.get("regularMarketPrice"), float("nan"))
        if symbol and price == price and price > 0:
            mapped[symbol] = {
                "price": round(price, 6),
                "bid": _safe_float(row.get("bid"), price),
                "ask": _safe_float(row.get("ask"), price),
                "change_pct_24h": _safe_float(row.get("regularMarketChangePercent"), 0.0),
                "trend_score": round(_safe_float(row.get("regularMarketChangePercent"), 0.0) / 5.0, 4),
                "source": "yahoo_quote",
            }
    return mapped


def _refresh_crypto_quotes(symbols: list[str]) -> Dict[str, Dict[str, Any]]:
    mapped: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        clean = str(symbol or "").strip().upper()
        exchange_symbol = CRYPTO_SYMBOL_MAP.get(clean)
        if not exchange_symbol:
            continue
        price_payload = _fetch_json(f"https://api.binance.com/api/v3/ticker/price?symbol={exchange_symbol}")
        stats_payload = _fetch_json(f"https://api.binance.com/api/v3/ticker/24hr?symbol={exchange_symbol}")
        price = _safe_float(((price_payload or {}) if isinstance(price_payload, dict) else {}).get("price"), float("nan"))
        if price != price or price <= 0:
            continue
        change_pct = _safe_float(((stats_payload or {}) if isinstance(stats_payload, dict) else {}).get("priceChangePercent"), 0.0)
        mapped[clean] = {
            "price": round(price, 6),
            "change_pct_24h": round(change_pct, 4),
            "trend_score": round(change_pct / 5.0, 4),
            "source": "binance_public",
        }
    return mapped


def _refresh_live_market_snapshot(state: Dict[str, Any], *, force: bool = False) -> None:
    snapshot = state.setdefault("market_snapshot", {})
    prices = snapshot.setdefault("prices", {})
    tracked = _tracked_symbols(state)
    now = _utc_now()
    now_iso = now.isoformat()
    updated_at = _parse_iso(snapshot.get("updated_at") or now_iso)
    if not force and (now - updated_at).total_seconds() < 12:
        return

    live_prices: Dict[str, Dict[str, Any]] = {}
    live_prices.update(_refresh_stock_quotes(tracked.get("stocks") or []))
    live_prices.update(_refresh_crypto_quotes(tracked.get("crypto") or []))
    feed_sources: Dict[str, Dict[str, Any]] = {}
    feed_freshness = {"stocks": {}, "crypto": {}}

    for market, symbols in tracked.items():
        for symbol in symbols:
            payload = live_prices.get(symbol)
            if payload:
                payload["timestamp"] = now_iso
                prices[symbol] = payload
            symbol_payload = prices.get(symbol)
            if not symbol_payload:
                continue
            timestamp = _parse_iso(symbol_payload.get("timestamp") or now_iso)
            freshness = max(0.0, (now - timestamp).total_seconds())
            provider_key = f"{str(symbol_payload.get('source') or 'cached').lower()}:{symbol}"
            feed_freshness.setdefault(market, {})[provider_key] = round(freshness, 2)
            feed_sources[symbol] = {
                "provider": str(symbol_payload.get("source") or "cached"),
                "timestamp": timestamp.isoformat(),
                "stale": freshness > (45 if market == "crypto" else 120),
            }

    snapshot["market"] = "mixed"
    snapshot["feed_freshness"] = feed_freshness
    snapshot["updated_at"] = now_iso
    snapshot["feed_sources"] = feed_sources
    status = state.setdefault("status", {})
    status["feed_freshness"] = deepcopy(feed_freshness)
    status["feed_sources"] = deepcopy(feed_sources)
    _persist_state(state)


def _touch_status(state: Dict[str, Any]) -> None:
    now = _utc_now()
    now_iso = now.isoformat()
    status = state.setdefault("status", {})
    status["mock_mode"] = True
    started_at = _parse_iso(status.get("started_at") or state.get("started_at") or now_iso)
    status["started_at"] = started_at.isoformat()
    status["timestamp"] = now_iso
    status["updated_at"] = now_iso
    status["last_heartbeat_at"] = now_iso
    status["last_heartbeat_ago"] = 0.0
    status["uptime_seconds"] = max(0, int((now - started_at).total_seconds()))
    for market in ("stocks", "crypto"):
        broker = ((status.get("broker_health") or {}).get(market) or {})
        broker["updated_at"] = now_iso
        broker["connected"] = True
        broker.setdefault("detail", "mock_account_ok")
        status.setdefault("broker_health", {})[market] = broker
    state["started_at"] = status["started_at"]
    _refresh_live_market_snapshot(state)


def _resolve_price(symbol: str) -> float:
    prices = ((STATE.get("market_snapshot") or {}).get("prices") or {})
    price_payload = prices.get(symbol) or {}
    for key in ("price", "last", "close", "mid"):
        value = _safe_float(price_payload.get(key), float("nan"))
        if value == value and value > 0:
            return value
    if symbol.endswith("-USD"):
        return 100.0
    return 50.0


def _update_portfolio_snapshot(state: Dict[str, Any]) -> None:
    positions = ((state.get("positions") or {}).get("items") or [])
    sim = ((state.get("status") or {}).get("simulation_portfolio") or {})
    cash = max(0.0, _safe_float(sim.get("cash"), _env_starting_balance()))
    grouped_positions = {"stocks": [], "crypto": []}
    total_positions_value = 0.0
    market_values = {"stocks": 0.0, "crypto": 0.0}
    for position in positions:
        symbol = str(position.get("symbol") or "").strip().upper()
        price = _resolve_price(symbol)
        quantity = max(0.0, _safe_float(position.get("quantity"), 0.0))
        position["current_price"] = price
        position["market_value"] = round(quantity * price, 2)
        total_positions_value += position["market_value"]
        market = str(position.get("market") or "stocks").strip().lower()
        if market not in grouped_positions:
            grouped_positions[market] = []
            market_values[market] = 0.0
        grouped_positions[market].append(position)
        market_values[market] += position["market_value"]
    portfolio_value = round(cash + total_positions_value, 2)
    sim["balance"] = portfolio_value
    sim["cash"] = round(cash, 2)
    sim["buying_power"] = round(cash, 2)
    sim["portfolio_value"] = portfolio_value
    state.setdefault("status", {})["simulation_portfolio"] = sim
    state["portfolio"] = {
        "simulation_mode": True,
        "mock_mode": True,
        "total_equity": portfolio_value,
        "simulation_portfolio": deepcopy(sim),
        "markets": {
            market: {"positions": rows}
            for market, rows in grouped_positions.items()
        },
    }
    state.setdefault("status", {})["equity"] = {
        "stocks": round(market_values.get("stocks", 0.0), 2),
        "crypto": round(market_values.get("crypto", 0.0), 2),
    }
    state.setdefault("recent_trades", {})["trades"] = list(state.get("recent_trades", {}).get("items") or [])


def _filter_symbol_rows(payload: Dict[str, Any], symbols: list[str]) -> Dict[str, Any]:
    if not symbols:
        return payload
    requested = {item.strip().upper() for item in symbols if item.strip()}
    filtered = dict(payload)
    rows = payload.get("symbols")
    if isinstance(rows, list):
        filtered["symbols"] = [row for row in rows if str((row or {}).get("symbol") or "").upper() in requested]
    return filtered


def _sector_for_symbol(symbol: str, market: str) -> str:
    upper = str(symbol or "").upper()
    lower_market = str(market or "").lower()
    if lower_market == "crypto":
        if upper.startswith("BTC"):
            return "store_of_value"
        if upper.startswith("ETH") or upper.startswith("SOL"):
            return "layer1"
        return "crypto"
    if upper in {"AAPL", "MSFT", "GOOG", "AMZN"}:
        return "large_cap_tech"
    return "equities"


def _build_exposure_payload(state: Dict[str, Any], market: str = "crypto") -> Dict[str, Any]:
    target_market = str(market or "crypto").strip().lower() or "crypto"
    portfolio = state.get("portfolio") or {}
    markets = (portfolio.get("markets") or {})
    market_rows = ((markets.get(target_market) or {}).get("positions") or [])
    sim = ((state.get("status") or {}).get("simulation_portfolio") or {})
    portfolio_value = max(0.0, _safe_float(sim.get("portfolio_value"), portfolio.get("total_equity")))
    cash = max(0.0, _safe_float(sim.get("cash"), 0.0))
    holdings_value = round(sum(_safe_float(row.get("market_value"), 0.0) for row in market_rows), 2)
    gross_exposure_pct = round((holdings_value / portfolio_value) * 100.0, 4) if portfolio_value > 0 else 0.0

    positions = []
    max_weight = 0.0
    for row in market_rows:
        market_value = round(_safe_float(row.get("market_value"), 0.0), 2)
        weight_pct = round((market_value / portfolio_value) * 100.0, 4) if portfolio_value > 0 else 0.0
        effective_weight_pct = weight_pct
        max_weight = max(max_weight, effective_weight_pct)
        positions.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "market": target_market,
                "quantity": round(_safe_float(row.get("quantity"), 0.0), 8),
                "price": round(_safe_float(row.get("current_price"), row.get("avg_entry_price")), 6),
                "market_value": market_value,
                "weight_pct": weight_pct,
                "effective_weight_pct": effective_weight_pct,
                "heat_pct": effective_weight_pct,
                "sector": _sector_for_symbol(str(row.get("symbol") or ""), target_market),
            }
        )

    return {
        "market": target_market,
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(cash, 2),
        "holdings_value": holdings_value,
        "gross_exposure_pct": gross_exposure_pct,
        "heat_score": round(max_weight, 4),
        "total_effective_heat_pct": gross_exposure_pct,
        "max_single_position_pct": 20.0,
        "max_total_gross_exposure_pct": 100.0,
        "simulation_mode": True,
        "active_markets": list(((state.get("status") or {}).get("active_markets") or [])),
        "positions": positions,
        "mock_mode": True,
        "updated_at": (state.get("status") or {}).get("updated_at") or _utc_now_iso(),
    }


def _list_strategies(state: Dict[str, Any]) -> Dict[str, Any]:
    status = state.get("status") or {}
    registry = status.get("strategy_registry") or {}
    controls = status.get("strategy_controls") or {}
    items = []
    for market in ("crypto", "stocks"):
        for item in registry.get(market) or []:
            strategy_name = str(item.get("strategy_name") or item.get("name") or "").strip()
            if not strategy_name:
                continue
            params = dict(item.get("params") or {})
            symbols = [str(symbol or "").strip().upper() for symbol in (item.get("symbols") or params.get("trade_symbols") or []) if str(symbol or "").strip()]
            last_update = params.get("updated_at") or status.get("updated_at")
            items.append(
                {
                    "strategy_key": strategy_name,
                    "strategy_name": strategy_name,
                    "market": market,
                    "enabled": bool((controls.get(market) or {}).get(strategy_name, item.get("enabled", True))),
                    "symbols": symbols,
                    "params": params,
                    "symbol_count": len(symbols),
                    "last_update": last_update,
                }
            )
    return {"items": items, "mock_mode": True, "updated_at": status.get("updated_at")}


def _upsert_strategy_locked(payload: Dict[str, Any]) -> Dict[str, Any]:
    status = STATE.setdefault("status", {})
    registry = status.setdefault("strategy_registry", deepcopy(DEFAULT_STRATEGY_REGISTRY))
    controls = status.setdefault("strategy_controls", {"stocks": {"StockMomentumStrategy": True}, "crypto": {"MomentumStrategy": True}})
    market = str(payload.get("market") or "crypto").strip().lower()
    strategy_name = str(payload.get("strategy_name") or payload.get("strategy_key") or "").strip() or "MomentumStrategy"
    params = dict(payload.get("params") or {})
    symbols = payload.get("symbols")
    if symbols is None:
        symbols = params.get("trade_symbols")
    if isinstance(symbols, list):
        clean_symbols = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    else:
        clean_symbols = []
    rows = registry.setdefault(market, [])
    existing = next((row for row in rows if str(row.get("strategy_name") or "").strip() == strategy_name), None)
    enabled = bool(payload.get("enabled")) if payload.get("enabled") is not None else bool((existing or {}).get("enabled", True))
    if existing is None:
        existing = {"strategy_name": strategy_name, "symbols": clean_symbols, "enabled": enabled, "params": {}}
        rows.append(existing)
    if clean_symbols:
        existing["symbols"] = clean_symbols
    existing["enabled"] = enabled
    merged_params = dict(existing.get("params") or {})
    merged_params.update(params)
    if clean_symbols:
        merged_params["trade_symbols"] = clean_symbols
    merged_params["updated_at"] = _utc_now_iso()
    existing["params"] = merged_params
    controls.setdefault(market, {})[strategy_name] = enabled
    symbol_controls = status.setdefault("symbol_controls", {}).setdefault(market, {})
    for symbol in clean_symbols:
        symbol_controls[symbol] = True
    _persist_state(STATE)
    return {"status": "updated", "strategy": existing, "mock_mode": True}


def _get_watchlists_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    payload = deepcopy(state.get("watchlists") or {"stocks": [], "crypto": []})
    return {
        "watchlists": payload,
        "strategy_candidates": list(state.get("strategy_candidates") or []),
        "mock_mode": True,
        "updated_at": ((state.get("status") or {}).get("updated_at") or _utc_now_iso()),
    }


def _update_watchlists_locked(payload: Dict[str, Any]) -> Dict[str, Any]:
    watchlists = STATE.setdefault("watchlists", {"stocks": [], "crypto": []})
    for market in ("stocks", "crypto"):
        if market in payload and isinstance(payload.get(market), list):
            watchlists[market] = [str(symbol or "").strip().upper() for symbol in payload.get(market) or [] if str(symbol or "").strip()]
    _persist_state(STATE)
    return _get_watchlists_payload(STATE)


def _queue_strategy_candidate_locked(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    market = str(payload.get("market") or "crypto").strip().lower() or "crypto"
    strategy_key = str(payload.get("strategy_key") or "").strip() or None
    if not symbol:
        return {"status": "rejected", "error": "symbol_required"}
    candidates = STATE.setdefault("strategy_candidates", [])
    existing = next((row for row in candidates if str(row.get("symbol") or "").upper() == symbol and str(row.get("market") or "").lower() == market and str(row.get("status") or "") == "pending"), None)
    if existing:
        return {"status": "queued", "item": existing, "mock_mode": True}
    item = {
        "id": f"candidate-{uuid.uuid4().hex[:10]}",
        "symbol": symbol,
        "market": market,
        "strategy_key": strategy_key,
        "status": "pending",
        "summary": str(payload.get("summary") or "").strip(),
        "analysis": payload.get("analysis") or {},
        "created_at": _utc_now_iso(),
    }
    candidates.insert(0, item)
    del candidates[100:]
    _persist_state(STATE)
    return {"status": "queued", "item": item, "mock_mode": True}


def _execute_trade_intent_locked(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    market = str(payload.get("market") or ("crypto" if symbol.endswith("-USD") else "stocks")).strip().lower()
    side = str(payload.get("side") or "buy").strip().lower()
    if not symbol:
        return {"status": "rejected", "error": "symbol_required"}

    sim = STATE.setdefault("status", {}).setdefault("simulation_portfolio", {})
    starting_balance = max(10.0, _safe_float(sim.get("starting_balance"), _env_starting_balance()))
    cash = max(0.0, _safe_float(sim.get("cash"), starting_balance))
    price = _resolve_price(symbol)
    size_hint = _safe_float(payload.get("size_hint"), 0.05)
    positions = STATE.setdefault("positions", {}).setdefault("items", [])
    existing = next((row for row in positions if str(row.get("symbol") or "").upper() == symbol and str(row.get("market") or "").lower() == market), None)

    if 0 < size_hint <= 1.0:
        target_notional = round(starting_balance * size_hint, 2)
    else:
        target_notional = round(size_hint if size_hint > 1.0 else starting_balance * 0.05, 2)

    now_iso = _utc_now_iso()
    trade_id = f"mock-trade-{uuid.uuid4().hex[:12]}"
    trade_payload: Dict[str, Any] = {
        "trade_id": trade_id,
        "id": trade_id,
        "market": market,
        "symbol": symbol,
        "side": side,
        "price": price,
        "source": "mock_simulation",
        "created_at": now_iso,
        "opened_at": now_iso,
        "timestamp": now_iso,
        "thesis": str(payload.get("thesis") or "").strip(),
        "confidence": _safe_float(payload.get("confidence"), 0.0),
    }

    if side == "sell":
        if not existing:
            return {"status": "rejected", "error": "no_position_to_sell", "received": payload}
        held_qty = max(0.0, _safe_float(existing.get("quantity"), 0.0))
        if held_qty <= 0:
            return {"status": "rejected", "error": "empty_position", "received": payload}
        requested_qty = target_notional / price if target_notional > 0 else held_qty
        quantity = min(held_qty, max(requested_qty, held_qty * 0.25))
        avg_entry = _safe_float(existing.get("avg_entry_price"), price)
        proceeds = round(quantity * price, 2)
        pnl = round((price - avg_entry) * quantity, 2)
        existing["quantity"] = round(max(0.0, held_qty - quantity), 8)
        cash = round(cash + proceeds, 2)
        sim["cash"] = cash
        trade_payload.update(
            {
                "quantity": round(quantity, 8),
                "notional": proceeds,
                "status": "filled",
                "closed_at": now_iso,
                "realized_pnl": pnl,
                "pnl": pnl,
                "avg_entry_price": avg_entry,
                "avg_exit_price": price,
            }
        )
        if existing["quantity"] <= 0:
            positions.remove(existing)
    else:
        notional = min(round(max(25.0, target_notional), 2), cash)
        if notional <= 0:
            return {"status": "rejected", "error": "insufficient_cash", "received": payload}
        quantity = round(notional / price, 8)
        cash = round(cash - notional, 2)
        sim["cash"] = cash
        if existing:
            held_qty = max(0.0, _safe_float(existing.get("quantity"), 0.0))
            held_notional = held_qty * _safe_float(existing.get("avg_entry_price"), price)
            new_qty = held_qty + quantity
            existing["quantity"] = round(new_qty, 8)
            existing["avg_entry_price"] = round((held_notional + notional) / max(new_qty, 1e-9), 6)
            existing["updated_at"] = now_iso
        else:
            positions.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "quantity": quantity,
                    "avg_entry_price": round(price, 6),
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
        trade_payload.update(
            {
                "quantity": quantity,
                "notional": notional,
                "status": "filled",
                "realized_pnl": 0.0,
                "pnl": 0.0,
                "avg_entry_price": price,
            }
        )

    _update_portfolio_snapshot(STATE)
    recent_trades = STATE.setdefault("recent_trades", {}).setdefault("items", [])
    recent_trades.insert(0, trade_payload)
    del recent_trades[50:]
    STATE.setdefault("recent_trades", {})["trades"] = list(recent_trades)
    STATE.setdefault("trade_lookup", {})[trade_id] = trade_payload
    STATE.setdefault("submitted_trade_intents", []).append(payload)
    _touch_status(STATE)
    _persist_state(STATE)
    return {
        "status": "submitted",
        "mock_mode": True,
        "trade_intent_id": f"intent-{len(STATE.get('submitted_trade_intents') or [])}",
        "execution_status": "filled",
        "trade": trade_payload,
        "portfolio": STATE.get("portfolio") or {},
    }


def _reset_simulation_locked(starting_balance: Optional[float], actor: str) -> Dict[str, Any]:
    research = list(STATE.get("research") or [])
    journal = list(STATE.get("journal") or [])
    next_state = _default_state(starting_balance=starting_balance)
    next_state["research"] = research
    next_state["journal"] = journal
    next_state.setdefault("status", {})["last_reset_by"] = actor
    next_state.setdefault("status", {})["last_reset_at"] = _utc_now_iso()
    STATE.clear()
    STATE.update(next_state)
    _persist_state(STATE)
    return {
        "status": "reset",
        "mock_mode": True,
        "actor": actor,
        "starting_balance": ((STATE.get("status") or {}).get("simulation_portfolio") or {}).get("starting_balance"),
        "portfolio": STATE.get("portfolio") or {},
    }


def _json(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        with STATE_LOCK:
            _touch_status(STATE)
            _update_portfolio_snapshot(STATE)
            if path == "/api/v1/ops/health":
                return _json(self, 200, {"ok": True, "mock_mode": True})
            if path == "/api/v1/ops/status":
                return _json(self, 200, deepcopy(STATE["status"]))
            if path == "/api/v1/ops/portfolio":
                return _json(self, 200, deepcopy(STATE["portfolio"]))
            if path == "/api/v1/ops/positions":
                market = (qs.get("market") or [None])[0]
                items = deepcopy((STATE["positions"] or {}).get("items") or [])
                if market:
                    items = [row for row in items if str((row or {}).get("market") or "").lower() == str(market).lower()]
                return _json(self, 200, {"items": items, "mock_mode": True})
            if path == "/api/v1/ops/incidents":
                payload = deepcopy(STATE["incidents"])
                payload["mock_mode"] = True
                return _json(self, 200, payload)
            if path == "/api/v1/ops/market-snapshot":
                payload = deepcopy(STATE["market_snapshot"])
                symbols = ((qs.get("symbols") or [""])[0]).split(",")
                if any(str(symbol).strip() for symbol in symbols):
                    payload["prices"] = {
                        str(symbol).strip().upper(): row
                        for symbol, row in (payload.get("prices") or {}).items()
                        if str(symbol).strip().upper() in {str(item).strip().upper() for item in symbols if str(item).strip()}
                    }
                payload["mock_mode"] = True
                return _json(self, 200, payload)
            if path == "/api/v1/ops/strategies":
                return _json(self, 200, _list_strategies(STATE))
            if path == "/api/v1/ops/sentiment":
                symbols = ((qs.get("symbols") or [""])[0]).split(",")
                payload = _filter_symbol_rows(deepcopy(STATE["sentiment"]), symbols)
                payload["mock_mode"] = True
                return _json(self, 200, payload)
            if path == "/api/v1/ops/confluence":
                symbols = ((qs.get("symbols") or [""])[0]).split(",")
                payload = _filter_symbol_rows(deepcopy(STATE["confluence"]), symbols)
                payload["mock_mode"] = True
                return _json(self, 200, payload)
            if path == "/api/v1/ops/event-risk":
                symbols = ((qs.get("symbols") or [""])[0]).split(",")
                payload = _filter_symbol_rows(deepcopy(STATE["event_risk"]), symbols)
                payload["mock_mode"] = True
                return _json(self, 200, payload)
            if path == "/api/v1/ops/exposure":
                market = (qs.get("market") or ["crypto"])[0]
                return _json(self, 200, _build_exposure_payload(STATE, market))
            if path == "/api/v1/ops/override/status":
                payload = deepcopy(STATE["override"])
                payload["mock_mode"] = True
                return _json(self, 200, payload)
            if path == "/api/v1/ops/trades/recent":
                payload = deepcopy(STATE["recent_trades"])
                payload["mock_mode"] = True
                return _json(self, 200, payload)
            if path == "/api/v1/ops/research":
                items = list(STATE.get("research") or [])
                market = (qs.get("market") or [None])[0]
                if market:
                    items = [row for row in items if str((row or {}).get("market") or "").lower() == str(market).lower()]
                return _json(self, 200, {"items": items, "mock_mode": True})
            if path == "/api/v1/ops/journal":
                limit = max(1, min(int((qs.get("limit") or [50])[0]), 200))
                items = list(STATE.get("journal") or [])[:limit]
                return _json(self, 200, {"items": items, "mock_mode": True})
            if path == "/api/v1/ops/watchlist":
                return _json(self, 200, _get_watchlists_payload(STATE))
            if path.startswith("/api/v1/ops/trades/"):
                trade_id = path.rsplit("/", 1)[-1]
                trade = deepcopy((STATE["trade_lookup"] or {}).get(trade_id))
                if not trade:
                    return _json(self, 404, {"error": "trade_not_found"})
                return _json(self, 200, trade)
            if path == "/__scenario":
                return _json(self, 200, deepcopy(STATE))

        return _json(self, 404, {"error": "not_found", "path": path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json()

        with STATE_LOCK:
            _touch_status(STATE)
            if path == "/api/v1/ops/trade-intents":
                return _json(self, 200, _execute_trade_intent_locked(payload))
            if path == "/api/v1/ops/research":
                item = dict(payload or {})
                item["created_at"] = _utc_now_iso()
                item["mock_mode"] = True
                STATE.setdefault("research", []).insert(0, item)
                _persist_state(STATE)
                return _json(self, 200, {"status": "stored", "mock_mode": True})
            if path == "/api/v1/ops/journal":
                item = dict(payload or {})
                item["created_at"] = _utc_now_iso()
                item["mock_mode"] = True
                STATE.setdefault("journal", []).insert(0, item)
                _persist_state(STATE)
                return _json(self, 200, {"status": "stored", "mock_mode": True})
            if path == "/api/v1/ops/pause":
                STATE["status"]["paused"] = True
                STATE["status"]["pause_reason"] = str(payload.get("reason") or "manual")
                _persist_state(STATE)
                return _json(self, 200, {"status": "paused", "mock_mode": True})
            if path == "/api/v1/ops/resume":
                STATE["status"]["paused"] = False
                STATE["status"]["pause_reason"] = ""
                _persist_state(STATE)
                return _json(self, 200, {"status": "running", "mock_mode": True})
            if path == "/api/v1/ops/risk":
                return _json(self, 200, {"status": "updated", "payload": payload, "mock_mode": True})
            if path == "/api/v1/ops/strategy":
                return _json(self, 200, _upsert_strategy_locked(payload))
            if path == "/api/v1/ops/override":
                now = _utc_now()
                ttl_minutes = int(payload.get("ttl_minutes") or 15)
                STATE["override"] = {
                    "enabled": True,
                    "actor": str(payload.get("actor") or "operator"),
                    "reason": str(payload.get("reason") or "manual override"),
                    "activated_at": now.isoformat(),
                    "expires_at": (now + timedelta(minutes=ttl_minutes)).isoformat(),
                    "ttl_seconds": ttl_minutes * 60,
                    "allowed_controls": [str(item).strip() for item in (payload.get("allowed_controls") or []) if str(item).strip()],
                    "last_cleared_at": (STATE.get("override") or {}).get("last_cleared_at"),
                    "mock_mode": True,
                }
                _persist_state(STATE)
                return _json(self, 200, deepcopy(STATE["override"]))
            if path == "/api/v1/ops/override/clear":
                previous = dict(STATE.get("override") or {})
                STATE["override"] = {
                    "enabled": False,
                    "actor": str(payload.get("actor") or ""),
                    "reason": str(payload.get("reason") or "manual clear"),
                    "activated_at": previous.get("activated_at"),
                    "expires_at": None,
                    "ttl_seconds": 0,
                    "allowed_controls": previous.get("allowed_controls") or [],
                    "last_cleared_at": _utc_now_iso(),
                    "mock_mode": True,
                }
                _persist_state(STATE)
                return _json(self, 200, deepcopy(STATE["override"]))
            if path.startswith("/api/v1/ops/incidents/") and path.endswith("/ack"):
                incident_id = path.split("/")[-2]
                return _json(self, 200, {"status": "acknowledged", "incident_id": incident_id, "mock_mode": True})
            if path.startswith("/api/v1/ops/incidents/") and path.endswith("/resolve"):
                incident_id = path.split("/")[-2]
                return _json(self, 200, {"status": "resolved", "incident_id": incident_id, "mock_mode": True})
            if path == "/api/v1/ops/simulation/reset":
                return _json(
                    self,
                    200,
                    _reset_simulation_locked(
                        starting_balance=_safe_float(payload.get("starting_balance"), _env_starting_balance()),
                        actor=str(payload.get("actor") or "operator"),
                    ),
                )
            if path == "/api/v1/ops/watchlist":
                return _json(self, 200, _update_watchlists_locked(payload))
            if path == "/api/v1/ops/strategy-candidates":
                return _json(self, 200, _queue_strategy_candidate_locked(payload))
            if path == "/api/v1/ops/refresh-feeds":
                _refresh_live_market_snapshot(STATE, force=True)
                _update_portfolio_snapshot(STATE)
                _persist_state(STATE)
                return _json(self, 200, {"status": "refreshed", "snapshot": deepcopy(STATE.get("market_snapshot") or {}), "mock_mode": True})
            if path == "/__scenario/reset":
                STATE.clear()
                STATE.update(_default_state())
                _persist_state(STATE)
                return _json(self, 200, deepcopy(STATE))
            if path == "/__scenario":
                for key, value in (payload or {}).items():
                    STATE[key] = _merge(STATE.get(key), value)
                _touch_status(STATE)
                _update_portfolio_snapshot(STATE)
                _persist_state(STATE)
                return _json(self, 200, deepcopy(STATE))

        return _json(self, 404, {"error": "not_found", "path": path})


if __name__ == "__main__":
    host = os.getenv("MOCK_LYSARA_HOST", "127.0.0.1")
    port = int(os.getenv("MOCK_LYSARA_PORT", "18792"))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Mock Lysara ops server listening on http://{host}:{port}")
    httpd.serve_forever()

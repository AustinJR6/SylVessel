#!/usr/bin/env python
from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _future(hours: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _default_state() -> Dict[str, Any]:
    now = _utc_now()
    return {
        "status": {
            "paused": False,
            "pause_reason": "",
            "simulation_mode": True,
            "live_trading_enabled": False,
            "autonomous_mode": True,
            "active_markets": ["stocks", "crypto"],
            "regime": "neutral",
            "equity": {"stocks": 1000.0, "crypto": 1000.0},
            "timestamp": now,
            "updated_at": now,
            "last_heartbeat_at": now,
            "uptime_seconds": 120,
            "last_heartbeat_ago": 5.0,
            "feed_freshness": {
                "stocks": {"alpaca_poll:AAPL": 2.0, "alpaca_poll:MSFT": 2.5},
                "crypto": {"binance_public_poll:BTC-USD": 1.5, "binance_public_poll:ETH-USD": 1.8},
            },
            "broker_health": {
                "stocks": {"connected": True, "detail": "account_ok", "updated_at": now},
                "crypto": {"connected": True, "detail": "account_ok", "updated_at": now},
            },
            "strategy_registry": {"stocks": [], "crypto": []},
            "symbol_controls": {"stocks": {"AAPL": True}, "crypto": {"BTC-USD": True, "ETH-USD": True}},
            "strategy_controls": {"stocks": {}, "crypto": {"MomentumStrategy": True}},
            "risk_managers": {"stocks": 1, "crypto": 1},
            "simulation_portfolio": {
                "currency": "USD",
                "starting_balance": 1000.0,
                "balance": 1000.0,
                "cash": 1000.0,
                "buying_power": 1000.0,
                "portfolio_value": 1000.0,
            },
        },
        "portfolio": {
            "total_equity": 20000.0,
        },
        "positions": {
            "items": [],
        },
        "incidents": {
            "items": [],
        },
        "market_snapshot": {
            "market": "stocks",
            "prices": {
                "AAPL": {"change_pct_24h": 1.2, "trend_score": 1.0},
                "MSFT": {"change_pct_24h": 0.8, "trend_score": 0.4},
                "BTC-USD": {"price": 69000.0, "source": "binance_public_poll"},
            },
            "feed_freshness": {
                "stocks": {"alpaca_poll:AAPL": 2.0},
                "crypto": {"binance_public_poll:BTC-USD": 1.5},
            },
        },
        "sentiment": {
            "updated_at": now,
            "configured_sources": ["newsapi", "reddit", "x"],
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
                    "sources": {
                        "newsapi": {"score": 0.15, "count": 12},
                        "reddit": {"score": 0.21, "count": 9},
                        "x": {"score": 0.18, "count": 22},
                    },
                    "updated_at": now,
                }
            ],
        },
        "confluence": {
            "updated_at": now,
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
            "updated_at": now,
            "configured_providers": ["coinmarketcal"],
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
                    "source": "coinmarketcal",
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
        "recent_trades": {
            "items": [],
        },
        "trade_lookup": {},
        "submitted_trade_intents": [],
        "journal": [],
        "research": [],
    }


DEFAULT_STATE: Dict[str, Any] = _default_state()
STATE: Dict[str, Any] = deepcopy(DEFAULT_STATE)


def _merge(existing: Any, incoming: Any) -> Any:
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = dict(existing)
        for key, value in incoming.items():
            merged[key] = _merge(existing.get(key), value)
        return merged
    return incoming


def _filter_symbol_rows(payload: Dict[str, Any], symbols: list[str]) -> Dict[str, Any]:
    if not symbols:
        return payload
    requested = {item.strip().upper() for item in symbols if item.strip()}
    filtered = dict(payload)
    rows = payload.get("symbols")
    if isinstance(rows, list):
        filtered["symbols"] = [row for row in rows if str((row or {}).get("symbol") or "").upper() in requested]
    return filtered


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

        if path == "/api/v1/ops/health":
            return _json(self, 200, {"ok": True})
        if path == "/api/v1/ops/status":
            return _json(self, 200, STATE["status"])
        if path == "/api/v1/ops/portfolio":
            return _json(self, 200, STATE["portfolio"])
        if path == "/api/v1/ops/positions":
            market = (qs.get("market") or [None])[0]
            items = STATE["positions"].get("items") or []
            if market:
                items = [row for row in items if str((row or {}).get("market") or "").lower() == str(market).lower()]
            return _json(self, 200, {"items": items})
        if path == "/api/v1/ops/incidents":
            return _json(self, 200, STATE["incidents"])
        if path == "/api/v1/ops/market-snapshot":
            return _json(self, 200, STATE["market_snapshot"])
        if path == "/api/v1/ops/sentiment":
            symbols = ((qs.get("symbols") or [""])[0]).split(",")
            return _json(self, 200, _filter_symbol_rows(STATE["sentiment"], symbols))
        if path == "/api/v1/ops/confluence":
            symbols = ((qs.get("symbols") or [""])[0]).split(",")
            return _json(self, 200, _filter_symbol_rows(STATE["confluence"], symbols))
        if path == "/api/v1/ops/event-risk":
            symbols = ((qs.get("symbols") or [""])[0]).split(",")
            return _json(self, 200, _filter_symbol_rows(STATE["event_risk"], symbols))
        if path == "/api/v1/ops/override/status":
            return _json(self, 200, STATE["override"])
        if path == "/api/v1/ops/trades/recent":
            return _json(self, 200, STATE["recent_trades"])
        if path.startswith("/api/v1/ops/trades/"):
            trade_id = path.rsplit("/", 1)[-1]
            trade = (STATE["trade_lookup"] or {}).get(trade_id)
            if not trade:
                return _json(self, 404, {"error": "trade_not_found"})
            return _json(self, 200, trade)
        if path == "/__scenario":
            return _json(self, 200, STATE)

        return _json(self, 404, {"error": "not_found", "path": path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json()

        if path == "/api/v1/ops/trade-intents":
            STATE["submitted_trade_intents"].append(payload)
            return _json(
                self,
                200,
                {
                    "status": "submitted",
                    "trade_intent_id": f"intent-{len(STATE['submitted_trade_intents'])}",
                    "received": payload,
                },
            )
        if path == "/api/v1/ops/research":
            STATE["research"].append(payload)
            return _json(self, 200, {"status": "stored"})
        if path == "/api/v1/ops/journal":
            STATE["journal"].append(payload)
            return _json(self, 200, {"status": "stored"})
        if path == "/api/v1/ops/pause":
            STATE["status"]["paused"] = True
            STATE["status"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            return _json(self, 200, {"status": "paused"})
        if path == "/api/v1/ops/resume":
            STATE["status"]["paused"] = False
            STATE["status"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            return _json(self, 200, {"status": "running"})
        if path == "/api/v1/ops/risk":
            return _json(self, 200, {"status": "updated", "payload": payload})
        if path == "/api/v1/ops/strategy":
            return _json(self, 200, {"status": "updated", "payload": payload})
        if path == "/api/v1/ops/override":
            now = datetime.now(timezone.utc)
            ttl_minutes = int(payload.get("ttl_minutes") or 15)
            STATE["override"] = {
                "enabled": True,
                "actor": str(payload.get("actor") or "operator"),
                "reason": str(payload.get("reason") or "manual override"),
                "activated_at": now.isoformat(),
                "expires_at": (now + timedelta(minutes=ttl_minutes)).isoformat(),
                "ttl_seconds": ttl_minutes * 60,
                "allowed_controls": [str(item).strip() for item in (payload.get("allowed_controls") or []) if str(item).strip()],
                "last_cleared_at": STATE.get("override", {}).get("last_cleared_at"),
            }
            return _json(self, 200, STATE["override"])
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
                "last_cleared_at": datetime.now(timezone.utc).isoformat(),
            }
            return _json(self, 200, STATE["override"])
        if path == "/__scenario/reset":
            STATE.clear()
            STATE.update(_default_state())
            return _json(self, 200, STATE)
        if path == "/__scenario":
            for key, value in (payload or {}).items():
                STATE[key] = _merge(STATE.get(key), value)
            if "status" in STATE and isinstance(STATE["status"], dict):
                STATE["status"]["updated_at"] = datetime.now(timezone.utc).isoformat()
                STATE["status"]["timestamp"] = STATE["status"]["updated_at"]
            return _json(self, 200, STATE)

        return _json(self, 404, {"error": "not_found", "path": path})


if __name__ == "__main__":
    host = os.getenv("MOCK_LYSARA_HOST", "127.0.0.1")
    port = int(os.getenv("MOCK_LYSARA_PORT", "18792"))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Mock Lysara ops server listening on http://{host}:{port}")
    httpd.serve_forever()

#!/usr/bin/env python
from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse


DEFAULT_STATE: Dict[str, Any] = {
    "status": {
        "paused": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
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
        },
    },
    "recent_trades": {
        "items": [],
    },
    "trade_lookup": {},
    "submitted_trade_intents": [],
    "journal": [],
    "research": [],
}

STATE: Dict[str, Any] = deepcopy(DEFAULT_STATE)


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
        if path == "/__scenario/reset":
            STATE.clear()
            STATE.update(deepcopy(DEFAULT_STATE))
            return _json(self, 200, STATE)
        if path == "/__scenario":
            for key, value in (payload or {}).items():
                STATE[key] = value
            if "status" in STATE and isinstance(STATE["status"], dict):
                STATE["status"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            return _json(self, 200, STATE)

        return _json(self, 404, {"error": "not_found", "path": path})


if __name__ == "__main__":
    host = os.getenv("MOCK_LYSARA_HOST", "127.0.0.1")
    port = int(os.getenv("MOCK_LYSARA_PORT", "18792"))
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Mock Lysara ops server listening on http://{host}:{port}")
    httpd.serve_forever()

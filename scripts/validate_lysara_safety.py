#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict
from urllib.error import HTTPError
from urllib.request import Request, urlopen


VESSEL_BASE = os.getenv("VESSEL_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
MOCK_BASE = os.getenv("MOCK_LYSARA_BASE_URL", "http://127.0.0.1:18792").rstrip("/")


def _request(method: str, url: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw or "{}")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {raw}") from exc


def vessel_get(path: str) -> Dict[str, Any]:
    return _request("GET", f"{VESSEL_BASE}{path}")


def vessel_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _request("POST", f"{VESSEL_BASE}{path}", payload)


def mock_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _request("POST", f"{MOCK_BASE}{path}", payload)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_guard_status_pause() -> None:
    mock_post("/__scenario/reset", {})
    mock_post("/__scenario", {"status": {"paused": True}})
    guard = vessel_get("/api/lysara/guard-status")
    assert_true(not guard["ok"], "guard should be blocked when trading is paused")
    assert_true("trading_paused" in guard["reasons"], "pause reason missing from guard status")


def test_sentiment_endpoint_health() -> None:
    mock_post("/__scenario/reset", {})
    payload = vessel_get("/api/lysara/sentiment?symbols=BTC-USD")
    rows = payload.get("symbols") or []
    assert_true(bool(payload.get("configured_sources")), "sentiment should report configured sources")
    assert_true(bool(rows), "sentiment should return at least one symbol row")
    assert_true(rows[0].get("symbol") == "BTC-USD", "sentiment should return the requested symbol")
    assert_true(rows[0].get("confidence") is not None, "sentiment row should include confidence")


def test_confluence_endpoint_health() -> None:
    mock_post("/__scenario/reset", {})
    payload = vessel_get("/api/lysara/confluence?symbols=BTC-USD")
    rows = payload.get("symbols") or []
    assert_true(bool(payload.get("timeframes")), "confluence should report timeframes")
    assert_true(bool(rows), "confluence should return at least one symbol row")
    assert_true(rows[0].get("symbol") == "BTC-USD", "confluence should return the requested symbol")
    assert_true(rows[0].get("confluence_score") is not None, "confluence row should include a score")


def test_event_risk_endpoint_health() -> None:
    mock_post("/__scenario/reset", {})
    payload = vessel_get("/api/lysara/event-risk?symbols=BTC-USD")
    rows = payload.get("symbols") or []
    assert_true(payload.get("lookahead_hours") is not None, "event risk should report lookahead window")
    assert_true(bool(rows), "event risk should return at least one symbol row")
    assert_true(rows[0].get("symbol") == "BTC-USD", "event risk should return the requested symbol")
    assert_true(rows[0].get("action") is not None, "event risk row should include an action")


def test_override_lifecycle() -> None:
    mock_post("/__scenario/reset", {})
    enabled = vessel_post(
        "/api/lysara/override",
        {
            "actor": "validator",
            "reason": "phase 5 validation",
            "ttl_minutes": 5,
            "allowed_controls": ["confidence_minimum", "event_risk_warning"],
        },
    )
    assert_true(enabled.get("enabled") is True, "override should enable")
    assert_true("confidence_minimum" in (enabled.get("allowed_controls") or []), "override should keep requested controls")

    status = vessel_get("/api/lysara/override/status")
    assert_true(status.get("enabled") is True, "override status should report enabled")
    assert_true(status.get("actor") == "validator", "override actor mismatch")

    cleared = vessel_post("/api/lysara/override/clear", {"actor": "validator", "reason": "validation complete"})
    assert_true(cleared.get("enabled") is False, "override should clear")
    assert_true(cleared.get("last_cleared_at"), "override clear should stamp last_cleared_at")


def test_hard_breaker_non_bypassable() -> None:
    mock_post("/__scenario/reset", {})
    vessel_post(
        "/api/lysara/override",
        {
            "actor": "validator",
            "reason": "check hard breakers",
            "ttl_minutes": 5,
            "allowed_controls": ["confidence_minimum", "event_risk_warning", "trade_cooldown"],
        },
    )
    mock_post("/__scenario", {"status": {"paused": True, "pause_reason": "hard breaker validation"}})
    guard = vessel_get("/api/lysara/guard-status")
    assert_true(not guard["ok"], "guard should remain blocked even with override enabled")
    assert_true("trading_paused" in guard["reasons"], "pause should remain a non-bypassable breaker")


def test_trade_approval_and_recheck() -> None:
    mock_post("/__scenario/reset", {})
    trade = vessel_post(
        "/api/lysara/trade-intents",
        {
            "market": "stocks",
            "symbol": "AAPL",
            "side": "buy",
            "thesis": "validation approval flow",
            "confidence": 0.7,
            "size_hint": 0.25,
            "actor": "validator",
            "source": "validation",
        },
    )
    assert_true(trade["status"] == "approval_required", "trade should require approval under large size")
    note = trade["approval_note"]
    mock_post(
        "/__scenario",
        {
            "positions": {
                "items": [
                    {"market": "stocks", "symbol": "AAPL", "notional": 18000},
                ]
            }
        },
    )
    approved = vessel_post(
        f"/sessions/proactive/notes/{note['note_id']}/approval",
        {"approved": True, "actor": "validator", "reason": "recheck validation"},
    )
    execution = approved.get("execution") or {}
    assert_true(execution.get("status") == "blocked_after_recheck", "approval should block when conditions changed")


def test_trade_close_reconciliation() -> None:
    mock_post("/__scenario/reset", {})
    mock_post(
        "/__scenario",
        {
            "trade_lookup": {
                "trade-123": {
                    "trade_id": "trade-123",
                    "market": "stocks",
                    "symbol": "MSFT",
                    "realized_pnl": 321.5,
                    "entry_price": 100.0,
                    "exit_price": 110.0,
                    "quantity": 10,
                    "fees": 1.25,
                    "closed_at": "2026-03-27T12:00:00+00:00",
                }
            }
        },
    )
    result = vessel_post(
        "/api/lysara/trade-close",
        {
            "trade_id": "trade-123",
            "pnl": -9999,
            "symbol": "WRONG",
        },
    )
    event = result["event"]
    assert_true(abs(float(event["pnl"]) - 321.5) < 0.001, "trade close should use reconciled PnL")
    assert_true(event["symbol"] == "MSFT", "trade close should use reconciled symbol")


def main() -> int:
    tests = [
        ("guard_status_pause", test_guard_status_pause),
        ("sentiment_endpoint_health", test_sentiment_endpoint_health),
        ("confluence_endpoint_health", test_confluence_endpoint_health),
        ("event_risk_endpoint_health", test_event_risk_endpoint_health),
        ("override_lifecycle", test_override_lifecycle),
        ("hard_breaker_non_bypassable", test_hard_breaker_non_bypassable),
        ("trade_approval_and_recheck", test_trade_approval_and_recheck),
        ("trade_close_reconciliation", test_trade_close_reconciliation),
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"FAIL {name}: {exc}")
    if failures:
        print(json.dumps({"ok": False, "failures": failures}, indent=2))
        return 1
    print(json.dumps({"ok": True, "tests": [name for name, _ in tests]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

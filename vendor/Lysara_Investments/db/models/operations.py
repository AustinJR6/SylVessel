from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any


def create_operations_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            actor TEXT NOT NULL,
            event_type TEXT NOT NULL,
            target TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'applied',
            details_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            kind TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            market TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            details_json TEXT NOT NULL DEFAULT '{}',
            acknowledged_at TEXT,
            resolved_at TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS research_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            market TEXT NOT NULL,
            symbol TEXT,
            summary TEXT NOT NULL,
            bullish_factors_json TEXT NOT NULL DEFAULT '[]',
            bearish_factors_json TEXT NOT NULL DEFAULT '[]',
            confidence REAL NOT NULL DEFAULT 0.0,
            horizon TEXT NOT NULL DEFAULT 'intraday',
            sources_json TEXT NOT NULL DEFAULT '[]',
            stale_after TEXT,
            actor TEXT NOT NULL DEFAULT 'system'
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            mode TEXT NOT NULL,
            action TEXT NOT NULL,
            status TEXT NOT NULL,
            market TEXT,
            symbol TEXT,
            summary TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}',
            trade_intent_id INTEGER
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_intents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            actor TEXT NOT NULL,
            source TEXT NOT NULL,
            market TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            thesis TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            size_hint REAL,
            time_horizon TEXT NOT NULL DEFAULT 'intraday',
            status TEXT NOT NULL,
            dedupe_key TEXT,
            policy_result_json TEXT NOT NULL DEFAULT '{}',
            execution_result_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_intents_dedupe_key ON trade_intents(dedupe_key)"
    )
    conn.commit()


def _now() -> str:
    return datetime.utcnow().isoformat()


def _json(value: Any, default: Any) -> str:
    payload = default if value is None else value
    return json.dumps(payload)


def insert_audit_event(
    conn: sqlite3.Connection,
    *,
    actor: str,
    event_type: str,
    target: str,
    status: str = "applied",
    details: dict[str, Any] | None = None,
) -> int:
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO audit_events (timestamp, actor, event_type, target, status, details_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (_now(), actor, event_type, target, status, _json(details, {})),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_incident(
    conn: sqlite3.Connection,
    *,
    kind: str,
    severity: str,
    message: str,
    market: str | None = None,
    status: str = "open",
    details: dict[str, Any] | None = None,
) -> int:
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO incidents (timestamp, kind, severity, message, market, status, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (_now(), kind, severity, message, market, status, _json(details, {})),
    )
    conn.commit()
    return int(cursor.lastrowid)


def acknowledge_incident(conn: sqlite3.Connection, incident_id: int) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE incidents
        SET status = CASE WHEN status = 'open' THEN 'acknowledged' ELSE status END,
            acknowledged_at = COALESCE(acknowledged_at, ?)
        WHERE id = ?
        """,
        (_now(), incident_id),
    )
    conn.commit()


def resolve_incident(conn: sqlite3.Connection, incident_id: int) -> None:
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE incidents
        SET status = 'resolved',
            resolved_at = ?,
            acknowledged_at = COALESCE(acknowledged_at, ?)
        WHERE id = ?
        """,
        (_now(), _now(), incident_id),
    )
    conn.commit()


def insert_research_note(
    conn: sqlite3.Connection,
    *,
    market: str,
    summary: str,
    symbol: str | None = None,
    bullish_factors: list[str] | None = None,
    bearish_factors: list[str] | None = None,
    confidence: float = 0.0,
    horizon: str = "intraday",
    sources: list[dict[str, Any]] | None = None,
    stale_after: str | None = None,
    actor: str = "system",
) -> int:
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO research_notes (
            timestamp, market, symbol, summary, bullish_factors_json, bearish_factors_json,
            confidence, horizon, sources_json, stale_after, actor
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _now(),
            market,
            symbol,
            summary,
            _json(bullish_factors, []),
            _json(bearish_factors, []),
            float(confidence),
            horizon,
            _json(sources, []),
            stale_after,
            actor,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_decision_journal(
    conn: sqlite3.Connection,
    *,
    mode: str,
    action: str,
    status: str,
    summary: str,
    market: str | None = None,
    symbol: str | None = None,
    details: dict[str, Any] | None = None,
    trade_intent_id: int | None = None,
) -> int:
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO decision_journal (timestamp, mode, action, status, market, symbol, summary, details_json, trade_intent_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (_now(), mode, action, status, market, symbol, summary, _json(details, {}), trade_intent_id),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_trade_intent(
    conn: sqlite3.Connection,
    *,
    actor: str,
    source: str,
    market: str,
    symbol: str,
    side: str,
    thesis: str,
    confidence: float,
    size_hint: float | None,
    time_horizon: str,
    status: str,
    dedupe_key: str | None = None,
    policy_result: dict[str, Any] | None = None,
    execution_result: dict[str, Any] | None = None,
) -> int:
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO trade_intents (
            id, timestamp, actor, source, market, symbol, side, thesis, confidence,
            size_hint, time_horizon, status, dedupe_key, policy_result_json, execution_result_json
        )
        VALUES (
            (SELECT id FROM trade_intents WHERE dedupe_key = ?),
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            dedupe_key,
            _now(),
            actor,
            source,
            market,
            symbol,
            side,
            thesis,
            float(confidence),
            size_hint,
            time_horizon,
            status,
            dedupe_key,
            _json(policy_result, {}),
            _json(execution_result, {}),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)

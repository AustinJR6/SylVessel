# dashboard/utils/data_access.py

"""Utility helpers to read dashboard data from the database and log files."""

from __future__ import annotations

import sqlite3
import json
import random
import datetime
from pathlib import Path
from typing import List, Dict

DB_PATH = "trades.db"
LOG_PATH = "logs/trading_bot.log"
SENTIMENT_FILE = Path("dashboard/data/sentiment_cache.json")


def _connect(db_path: str = DB_PATH):
    """Return sqlite connection or None if DB missing."""
    if not Path(db_path).is_file():
        return None
    return sqlite3.connect(db_path, check_same_thread=False)


def get_trade_history(limit: int = 50, db_path: str = DB_PATH) -> List[Dict]:
    conn = _connect(db_path)
    if not conn:
        return []
    cur = conn.cursor()
    cur.execute(
        """
        SELECT timestamp, symbol, side, quantity, price, profit_loss, reason, market
        FROM trades ORDER BY timestamp DESC LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    cols = [
        "timestamp",
        "symbol",
        "side",
        "quantity",
        "price",
        "pnl",
        "reason",
        "market",
    ]
    return [dict(zip(cols, r)) for r in rows]


def get_last_trade(db_path: str = DB_PATH):
    trades = get_trade_history(1, db_path)
    return trades[0] if trades else None


def get_last_trade_per_market(db_path: str = DB_PATH) -> Dict[str, Dict]:
    """Return the most recent trade for each market."""
    markets = ["crypto", "stocks", "forex"]
    result: Dict[str, Dict] = {}
    conn = _connect(db_path)
    if not conn:
        return result
    cur = conn.cursor()
    for m in markets:
        cur.execute(
            """
            SELECT timestamp, symbol, side, quantity, price, profit_loss, reason, market
            FROM trades WHERE market=? ORDER BY timestamp DESC LIMIT 1
            """,
            (m,),
        )
        row = cur.fetchone()
        if row:
            cols = [
                "timestamp",
                "symbol",
                "side",
                "quantity",
                "price",
                "pnl",
                "reason",
                "market",
            ]
            result[m] = dict(zip(cols, row))
    conn.close()
    return result


def get_equity(db_path: str = DB_PATH) -> float:
    conn = _connect(db_path)
    if not conn:
        return 0.0
    cur = conn.cursor()
    cur.execute(
        "SELECT total_equity FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1"
    )
    row = cur.fetchone()
    conn.close()
    return float(row[0]) if row else 0.0


def get_performance_metrics(db_path: str = DB_PATH) -> Dict:
    trades = get_trade_history(1000, db_path)
    closed = [t for t in trades if t["pnl"] is not None]
    wins = [t for t in closed if t["pnl"] > 0]
    win_rate = round(len(wins) / len(closed) * 100, 2) if closed else 0.0
    avg_return = round(sum(t["pnl"] for t in closed) / len(closed), 4) if closed else 0.0
    open_risk = round(
        sum(t["quantity"] * t["price"] for t in trades if t["pnl"] is None), 2
    )
    return {
        "win_rate": win_rate,
        "avg_return": avg_return,
        "open_risk": open_risk,
        "trade_count": len(closed),
    }


def get_equity_curve(limit: int = 500, db_path: str = DB_PATH) -> List[Dict]:
    conn = _connect(db_path)
    if not conn:
        return []
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, total_equity FROM equity_snapshots ORDER BY timestamp DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    data = [{"timestamp": r[0], "equity": float(r[1])} for r in rows]
    data.reverse()
    return data


def get_log_lines(limit: int = 200, log_path: str = LOG_PATH) -> List[str]:
    if not Path(log_path).is_file():
        return []
    with open(log_path, "r") as f:
        lines = f.readlines()[-limit:]
    return [l.rstrip() for l in lines]


def get_ai_thoughts(limit: int = 50, log_path: str = "logs/ai_decisions.log") -> List[Dict]:
    """Return recent AI strategist decisions."""
    path = Path(log_path)
    if not path.is_file():
        return []
    lines = path.read_text().strip().splitlines()[-limit:]
    entries = []
    for line in lines:
        try:
            ts_part, rest = line.split(" ", 1)
            ctx_str = rest.split("context=")[1].split(" decision=")[0]
            dec_str = rest.split("decision=")[1]
            decision = json.loads(dec_str)
            entries.append({
                "timestamp": ts_part,
                "action": decision.get("action"),
                "confidence": decision.get("confidence"),
                "reason": decision.get("reason"),
            })
        except Exception:
            continue
    return entries


def get_sentiment_data(file_path: Path = SENTIMENT_FILE) -> Dict:
    if file_path.is_file():
        try:
            return json.loads(file_path.read_text())
        except Exception:
            return {}
    return {}


def mock_trade_history(count: int = 10) -> List[Dict]:
    """Produce random mock trades for demo/simulation mode."""
    now = datetime.datetime.utcnow()
    trades = []
    for i in range(count):
        trades.append(
            {
                "timestamp": (now - datetime.timedelta(minutes=5 * i)).isoformat(),
                "symbol": "BTC-USD",
                "side": random.choice(["buy", "sell"]),
                "quantity": round(random.uniform(0.01, 0.05), 4),
                "price": round(30000 + random.uniform(-1000, 1000), 2),
                "pnl": round(random.uniform(-50, 50), 2),
                "reason": "sim",
                "market": "crypto",
            }
        )
    return trades


def get_last_agent_decision(log_path: str = "logs/agent_decisions.log") -> Dict:
    """Return the most recent agent decision."""
    path = Path(log_path)
    if not path.is_file():
        return {}
    try:
        lines = path.read_text().strip().splitlines()
        if not lines:
            return {}
        last = lines[-1]
        ts, rest = last.split(" ", 1)
        ctx = rest.split("context=")[1].split(" decision=")[0]
        dec = rest.split("decision=")[1]
        return {
            "timestamp": ts,
            "context": json.loads(ctx),
            "decision": json.loads(dec),
        }
    except Exception:
        return {}

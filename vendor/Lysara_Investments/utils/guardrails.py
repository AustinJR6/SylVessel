import os
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
import asyncio
from .notifications import send_slack_message
from .runtime_paths import env_or_runtime_path

async def log_live_trade(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    config: dict,
    market: str = "crypto",
    confidence: float | None = None,
    risk_pct: float | None = None,
):
    """Append live trade details to log file and send Slack alert."""
    log_line = f"{datetime.utcnow().isoformat()} {symbol} {side} {qty} @ {price}"
    log_path = env_or_runtime_path("TRADE_LOG_PATH", "logs", "trade_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(log_line + "\n")

    db_path = env_or_runtime_path("TRADE_LOG_DB_PATH", "trade_logs.db")
    try:
        db_path.parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL
            )"""
        )
        cur.execute(
            "INSERT INTO trades(timestamp, symbol, side, quantity, price) VALUES(?,?,?,?,?)",
            (datetime.utcnow().isoformat(), symbol, side, qty, price)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Trade DB log failed: {e}")

    webhook = config.get("api_keys", {}).get("slack_webhook")
    if webhook:
        msg = (
            "\ud83d\udea8 LIVE TRADE EXECUTED\n"
            f"Market: {market.capitalize()}\n"
            f"Action: {side.upper()} {qty} {symbol} @ ${price}"
        )
        if confidence is not None or risk_pct is not None:
            conf = f"{confidence:.2f}" if confidence is not None else "N/A"
            risk = f"{risk_pct}%" if risk_pct is not None else "N/A"
            msg += f"\nConfidence: {conf} | Risk: {risk}"
        await send_slack_message(webhook, msg)


def confirm_live_mode(simulation_mode: bool):
    """Prompt user before starting live trading."""
    if not simulation_mode:
        if os.getenv("ALLOW_LIVE_TRADING_STARTUP", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
            logging.warning("Live trading startup approved by ALLOW_LIVE_TRADING_STARTUP.")
            return
        if not sys.stdin.isatty():
            logging.error(
                "Live trading startup requires confirmation. Set ALLOW_LIVE_TRADING_STARTUP=true for non-interactive launches."
            )
            sys.exit(1)
        resp = input("[!] SIMULATION_MODE is OFF. Proceed with live trading? (y/N) ")
        if resp.strip().lower() != "y":
            logging.warning("Aborting launch.")
            sys.exit(0)


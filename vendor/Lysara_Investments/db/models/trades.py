# db/models/trades.py

import sqlite3
from datetime import datetime

def create_trades_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            profit_loss REAL,
            reason TEXT,
            market TEXT NOT NULL,
            conviction REAL,
            signal_profile TEXT
        )
    """)
    conn.commit()

def insert_trade(
    conn: sqlite3.Connection,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    profit_loss: float = None,
    reason: str = "",
    market: str = "crypto",
    conviction: float | None = None,
    signal_profile: str | None = None,
):
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO trades (timestamp, symbol, side, quantity, price, profit_loss, reason, market, conviction, signal_profile)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            symbol,
            side,
            quantity,
            price,
            profit_loss,
            reason,
            market,
            conviction,
            signal_profile,
        ),
    )
    conn.commit()

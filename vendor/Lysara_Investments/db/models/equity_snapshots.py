# db/models/equity_snapshots.py

import sqlite3
from datetime import datetime

def create_equity_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_equity REAL NOT NULL,
            market TEXT NOT NULL
        )
    """)
    conn.commit()

def insert_equity_snapshot(conn: sqlite3.Connection, equity: float, market: str):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO equity_snapshots (timestamp, total_equity, market)
        VALUES (?, ?, ?)
    """, (datetime.utcnow().isoformat(), equity, market))
    conn.commit()

# db/models/orders.py

import sqlite3
from datetime import datetime

def create_orders_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            order_type TEXT NOT NULL,
            status TEXT NOT NULL,
            market TEXT NOT NULL
        )
    """)
    conn.commit()

def insert_order(conn: sqlite3.Connection, symbol: str, side: str, quantity: float, price: float, order_type: str, status: str, market: str):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO orders (timestamp, symbol, side, quantity, price, order_type, status, market)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        symbol,
        side,
        quantity,
        price,
        order_type,
        status,
        market
    ))
    conn.commit()

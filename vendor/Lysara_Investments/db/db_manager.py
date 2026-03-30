# db/db_manager.py

import sqlite3
import logging
from db.models.trades import create_trades_table, insert_trade
from db.models.orders import create_orders_table, insert_order
from db.models.equity_snapshots import create_equity_table, insert_equity_snapshot
from db.models.operations import (
    acknowledge_incident,
    create_operations_tables,
    insert_audit_event,
    insert_decision_journal,
    insert_incident,
    insert_research_note,
    insert_trade_intent,
    resolve_incident,
)

class DatabaseManager:
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize_tables()

    def _initialize_tables(self):
        logging.info("Initializing database tables...")
        create_trades_table(self.conn)
        create_orders_table(self.conn)
        create_equity_table(self.conn)
        create_operations_tables(self.conn)

    def log_trade(self, **kwargs):
        try:
            insert_trade(self.conn, **kwargs)
            logging.info(f"Trade logged: {kwargs}")
        except Exception as e:
            logging.error(f"Failed to log trade: {e}")

    def log_order(self, **kwargs):
        try:
            insert_order(self.conn, **kwargs)
            logging.info(f"Order logged: {kwargs}")
        except Exception as e:
            logging.error(f"Failed to log order: {e}")

    def log_equity_snapshot(self, equity: float, market: str):
        try:
            insert_equity_snapshot(self.conn, equity, market)
            logging.info(f"Equity snapshot logged: {equity} ({market})")
        except Exception as e:
            logging.error(f"Failed to log equity snapshot: {e}")

    def close(self):
        self.conn.close()
        logging.info("Database connection closed.")

    def fetch_all(self, query: str, params: tuple = ()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        return [dict(row) for row in cur.fetchall()]

    def fetch_one(self, query: str, params: tuple = ()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def log_audit_event(self, **kwargs):
        try:
            return insert_audit_event(self.conn, **kwargs)
        except Exception as e:
            logging.error(f"Failed to log audit event: {e}")
            return None

    def log_incident(self, **kwargs):
        try:
            return insert_incident(self.conn, **kwargs)
        except Exception as e:
            logging.error(f"Failed to log incident: {e}")
            return None

    def acknowledge_incident(self, incident_id: int):
        try:
            acknowledge_incident(self.conn, incident_id)
        except Exception as e:
            logging.error(f"Failed to acknowledge incident {incident_id}: {e}")

    def resolve_incident(self, incident_id: int):
        try:
            resolve_incident(self.conn, incident_id)
        except Exception as e:
            logging.error(f"Failed to resolve incident {incident_id}: {e}")

    def log_research_note(self, **kwargs):
        try:
            return insert_research_note(self.conn, **kwargs)
        except Exception as e:
            logging.error(f"Failed to log research note: {e}")
            return None

    def log_decision_journal(self, **kwargs):
        try:
            return insert_decision_journal(self.conn, **kwargs)
        except Exception as e:
            logging.error(f"Failed to log decision journal: {e}")
            return None

    def log_trade_intent(self, **kwargs):
        try:
            return insert_trade_intent(self.conn, **kwargs)
        except Exception as e:
            logging.error(f"Failed to log trade intent: {e}")
            return None

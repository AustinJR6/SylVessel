"""
Sylana Vessel - Supabase Database Client
Provides connection to Supabase PostgreSQL with pgvector support.
All modules import get_connection() from here instead of using sqlite3.
"""

import os
import logging
import time

import psycopg2
import psycopg2.extras
from psycopg2 import extensions
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)

_connection = None


def get_connection():
    """
    Get or create a persistent connection to Supabase PostgreSQL.
    Uses SUPABASE_DB_URL from environment.
    Auto-reconnects if the connection was closed.
    """
    global _connection
    if _connection is not None and not _connection.closed:
        # Recover shared connection if a previous statement aborted the transaction.
        try:
            tx_status = _connection.get_transaction_status()
            if tx_status == extensions.TRANSACTION_STATUS_INERROR:
                _connection.rollback()
                logger.warning("Recovered Supabase connection from aborted transaction state")
        except Exception as e:
            logger.warning(f"Failed transaction-state recovery check: {e}")
        return _connection

    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError(
            "SUPABASE_DB_URL not set. Add it to .env.\n"
            "Format: postgresql://postgres.[ref]:[password]@[host]:5432/postgres"
        )

    # Retry up to 3 times for transient network issues
    for attempt in range(3):
        try:
            _connection = psycopg2.connect(db_url)
            _connection.autocommit = False
            register_vector(_connection)
            logger.info("Connected to Supabase PostgreSQL")
            return _connection
        except psycopg2.OperationalError as e:
            if attempt < 2:
                wait = (attempt + 1) * 2
                logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"Failed to connect to Supabase after 3 attempts: {e}")
                raise


def close_connection():
    """Close the persistent connection."""
    global _connection
    if _connection and not _connection.closed:
        _connection.close()
        _connection = None
        logger.info("Supabase connection closed")

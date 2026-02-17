"""
Sylana Vessel - Supabase Database Client
Provides connection to Supabase PostgreSQL with pgvector support.
All modules import get_connection() from here instead of using sqlite3.
"""

import os
import logging
import time
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
from psycopg2 import extensions
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)

_connection = None


def _get_sanitized_db_url() -> str:
    """Read SUPABASE_DB_URL and normalize common secret-manager formatting issues."""
    raw_value = os.getenv("SUPABASE_DB_URL", "")
    db_url = raw_value.strip().strip('"').strip("'")

    if not db_url:
        raise RuntimeError(
            "SUPABASE_DB_URL not set. Add it to .env.\n"
            "Format: postgresql://postgres.[ref]:[password]@[host]:5432/postgres"
        )

    if "\n" in db_url or "\r" in db_url:
        raise RuntimeError("SUPABASE_DB_URL contains newline characters; it must be a single line")

    parsed = urlparse(db_url)
    if parsed.netloc.count("@") > 1:
        raise RuntimeError(
            "SUPABASE_DB_URL appears to have an unescaped '@' in credentials. "
            "URL-encode your password before putting it in the connection URL."
        )

    if not parsed.hostname or not parsed.path or parsed.path == "/":
        raise RuntimeError(
            "SUPABASE_DB_URL is malformed. Expected: postgresql://user:pass@host:5432/postgres"
        )

    return db_url


def _log_connection_target(db_url: str) -> None:
    """Log safe connection target fields to aid cloud debugging."""
    parsed = urlparse(db_url)
    db_name = parsed.path.lstrip("/") or "<missing>"
    port = parsed.port or 5432
    logger.info(
        "Connecting to Supabase PostgreSQL host=%s port=%s db=%s",
        parsed.hostname,
        port,
        db_name,
    )


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

    db_url = _get_sanitized_db_url()
    _log_connection_target(db_url)

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

"""
Sylana Vessel - Supabase Database Client
Provides connection to Supabase PostgreSQL with pgvector support.
All modules import get_connection() from here instead of using sqlite3.
"""

import os
import logging
import time
from contextlib import contextmanager
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras
from psycopg2 import pool
from psycopg2 import extensions
from pgvector.psycopg2 import register_vector

logger = logging.getLogger(__name__)

_connection = None
_pool = None


def _discard_connection(reason: str = "") -> None:
    """Drop cached connection reference safely so a new one can be created."""
    global _connection
    if _connection is None:
        return
    try:
        if not _connection.closed:
            _connection.close()
    except Exception:
        pass
    finally:
        _connection = None
    if reason:
        logger.warning("Discarded Supabase connection: %s", reason)


def _connection_kwargs(db_url: str) -> dict:
    return {
        "dsn": db_url,
        "connect_timeout": 10,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 3,
        "application_name": "sylana-vessel",
    }


def init_connection_pool(minconn: int = 1, maxconn: int = 8):
    """Initialize pooled Supabase connections for concurrent requests."""
    global _pool
    if _pool is not None:
        return _pool

    db_url = _get_sanitized_db_url()
    _log_connection_target(db_url)
    env_min = int(os.getenv("DB_POOL_MIN", str(minconn or 1)))
    env_max = int(os.getenv("DB_POOL_MAX", str(maxconn or 8)))
    minconn = max(1, env_min)
    maxconn = max(minconn, env_max)

    try:
        _pool = pool.ThreadedConnectionPool(minconn, maxconn, **_connection_kwargs(db_url))
        logger.info("Initialized Supabase connection pool (min=%s max=%s)", minconn, maxconn)
        return _pool
    except Exception as e:
        logger.warning("Failed to initialize connection pool; falling back to singleton connection: %s", e)
        _pool = None
        return None


def get_pooled_connection():
    """
    Borrow a connection from the pool.
    Falls back to singleton connection when pool is unavailable.
    """
    global _pool
    if _pool is None:
        init_connection_pool()
    if _pool is None:
        return get_connection()

    conn = _pool.getconn()
    conn.autocommit = False
    register_vector(conn)
    return conn


def release_pooled_connection(conn):
    """Return pooled connection to pool; close if pool is unavailable."""
    global _pool
    if conn is None:
        return
    if _pool is None:
        try:
            if not conn.closed:
                conn.close()
        except Exception:
            pass
        return
    try:
        if conn.get_transaction_status() == extensions.TRANSACTION_STATUS_INERROR:
            conn.rollback()
    except Exception:
        pass
    _pool.putconn(conn)


@contextmanager
def pooled_cursor(commit: bool = False):
    """
    Context-managed cursor on a pooled connection.
    Use this in modular codepaths to avoid shared-connection contention.
    """
    conn = get_pooled_connection()
    cur = conn.cursor()
    try:
        yield cur
        if commit:
            conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        release_pooled_connection(conn)


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
            # Detect stale sockets that still report closed=0.
            with _connection.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return _connection
        except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
            _discard_connection(f"stale or closed socket detected ({e})")
        except Exception as e:
            logger.warning(f"Failed transaction-state recovery check: {e}")

    db_url = _get_sanitized_db_url()
    _log_connection_target(db_url)

    # Retry up to 3 times for transient network issues
    for attempt in range(3):
        try:
            _connection = psycopg2.connect(**_connection_kwargs(db_url))
            _connection.autocommit = False
            register_vector(_connection)
            logger.info("Connected to Supabase PostgreSQL")
            return _connection
        except psycopg2.OperationalError as e:
            _discard_connection(f"connect attempt failed ({e})")
            if attempt < 2:
                wait = (attempt + 1) * 2
                logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"Failed to connect to Supabase after 3 attempts: {e}")
                raise


def close_connection():
    """Close singleton connection and pool connections."""
    global _connection, _pool
    if _connection and not _connection.closed:
        _connection.close()
        _connection = None
        logger.info("Supabase connection closed")
    if _pool is not None:
        try:
            _pool.closeall()
            logger.info("Supabase connection pool closed")
        finally:
            _pool = None

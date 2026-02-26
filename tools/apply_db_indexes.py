"""
Memory search index migration for Sylana Vessel.

Applies three improvements to the Supabase `memories` table:
  1. fts_vector — generated tsvector column for full-text search
  2. GIN index on fts_vector — fast FTS queries
  3. Partial indexes on emotion, memory_type, significance_score (filtered
     to personality = 'sylana') for fast filtered queries

Safe to run multiple times — all statements use IF NOT EXISTS.

Usage:
  python tools/apply_db_indexes.py
"""

import sys
import logging
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_supabase_client():
    """
    Import memory/supabase_client.py directly, bypassing memory/__init__.py
    which would pull in openai/torch/etc. that aren't needed here.
    """
    # Load .env if present
    env_file = ROOT / ".env"
    if env_file.exists():
        import os
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

    spec = importlib.util.spec_from_file_location(
        "supabase_client", ROOT / "memory" / "supabase_client.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Each step is (description, sql).
# We run them individually so one failure doesn't abort the rest.
MIGRATIONS = [
    (
        "Add fts_vector generated column (full-text search)",
        """
        ALTER TABLE memories
        ADD COLUMN IF NOT EXISTS fts_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector(
                'english',
                coalesce(user_input, '') || ' ' ||
                coalesce(sylana_response, '') || ' ' ||
                coalesce(topic, '') || ' ' ||
                coalesce(emotion, '')
            )
        ) STORED;
        """,
    ),
    (
        "GIN index on fts_vector",
        "CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN(fts_vector);",
    ),
    (
        "Partial index: significance_score for sylana",
        """
        CREATE INDEX IF NOT EXISTS idx_memories_sylana_sig
        ON memories(significance_score DESC)
        WHERE personality = 'sylana';
        """,
    ),
    (
        "Partial index: emotion + significance for sylana",
        """
        CREATE INDEX IF NOT EXISTS idx_memories_sylana_emotion
        ON memories(emotion, significance_score DESC)
        WHERE personality = 'sylana';
        """,
    ),
    (
        "Partial index: memory_type + significance for sylana",
        """
        CREATE INDEX IF NOT EXISTS idx_memories_sylana_type
        ON memories(memory_type, significance_score DESC)
        WHERE personality = 'sylana';
        """,
    ),
]


def run():
    sc = _load_supabase_client()
    get_connection = sc.get_connection

    conn = get_connection()
    cur = conn.cursor()

    passed = 0
    failed = 0

    for description, sql in MIGRATIONS:
        try:
            logger.info(f"Applying: {description}")
            cur.execute(sql.strip())
            conn.commit()
            logger.info(f"  OK")
            passed += 1
        except Exception as e:
            conn.rollback()
            logger.error(f"  FAILED: {e}")
            failed += 1

    logger.info(f"\nDone — {passed} succeeded, {failed} failed.")
    if failed:
        logger.warning(
            "Some steps failed. If 'column fts_vector already exists' or "
            "'index already exists', those are safe to ignore."
        )
    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)

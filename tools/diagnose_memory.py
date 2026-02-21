"""
Memory diagnostic script for Sylana Vessel.

Connects to Supabase and runs targeted diagnostic queries to inspect
the state of the memories table — personality distribution, embedding
model coverage, conversation-linked sylana rows, top memories by
significance, and fts_vector population.

Usage:
  python tools/diagnose_memory.py
"""

import sys
import logging
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEP = "-" * 72


def _load_env():
    """Load .env from project root into os.environ, skipping already-set keys."""
    import os
    env_file = ROOT / ".env"
    if not env_file.exists():
        logger.warning(".env not found at %s", env_file)
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
    logger.info("Loaded .env from %s", env_file)


def _load_supabase_client():
    """Import memory/supabase_client.py directly, bypassing memory/__init__.py."""
    spec = importlib.util.spec_from_file_location(
        "supabase_client", ROOT / "memory" / "supabase_client.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run():
    _load_env()
    sc = _load_supabase_client()
    conn = sc.get_connection()
    cur = conn.cursor()

    # ------------------------------------------------------------------ #
    # (a) COUNT of memories by personality value (NULL + empty string)    #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("(a) COUNT of memories by personality value (including NULL and empty string)")
    print(SEP)
    cur.execute("""
        SELECT
            CASE
                WHEN personality IS NULL         THEN '<NULL>'
                WHEN personality = ''            THEN '<empty string>'
                ELSE personality
            END AS personality_bucket,
            COUNT(*) AS cnt
        FROM memories
        GROUP BY personality_bucket
        ORDER BY cnt DESC;
    """)
    rows = cur.fetchall()
    if rows:
        print(f"  {'personality':<30}  {'count':>8}")
        print(f"  {'-'*30}  {'-'*8}")
        for personality_bucket, cnt in rows:
            print(f"  {personality_bucket:<30}  {cnt:>8}")
    else:
        print("  (no rows returned)")

    # ------------------------------------------------------------------ #
    # (b) COUNT of memories by embedding_model value (including NULL)     #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("(b) COUNT of memories by embedding_model value (including NULL)")
    print(SEP)
    cur.execute("""
        SELECT
            CASE
                WHEN embedding_model IS NULL THEN '<NULL>'
                WHEN embedding_model = ''    THEN '<empty string>'
                ELSE embedding_model
            END AS model_bucket,
            COUNT(*) AS cnt
        FROM memories
        GROUP BY model_bucket
        ORDER BY cnt DESC;
    """)
    rows = cur.fetchall()
    if rows:
        print(f"  {'embedding_model':<40}  {'count':>8}")
        print(f"  {'-'*40}  {'-'*8}")
        for model_bucket, cnt in rows:
            print(f"  {model_bucket:<40}  {cnt:>8}")
    else:
        print("  (no rows returned)")

    # ------------------------------------------------------------------ #
    # (c) Memories where conversation_id IS NOT NULL AND personality='sylana' #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("(c) Memories with conversation_id IS NOT NULL AND personality='sylana'")
    print(SEP)
    cur.execute("""
        SELECT COUNT(*) AS cnt
        FROM memories
        WHERE conversation_id IS NOT NULL
          AND personality = 'sylana';
    """)
    (cnt,) = cur.fetchone()
    print(f"  Count: {cnt}")

    # ------------------------------------------------------------------ #
    # (d) Top 10 imported memories ordered by significance_score DESC     #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("(d) Top 10 imported memories by significance_score DESC")
    print("    Columns: id, conversation_title, emotion, personality, embedding_model,")
    print("             first 100 chars of user_input")
    print(SEP)
    cur.execute("""
        SELECT
            id,
            conversation_title,
            emotion,
            personality,
            embedding_model,
            LEFT(user_input, 100) AS user_input_preview,
            significance_score
        FROM memories
        ORDER BY significance_score DESC NULLS LAST
        LIMIT 10;
    """)
    rows = cur.fetchall()
    if rows:
        for i, (row_id, conv_title, emotion, personality, emb_model,
                user_input_preview, sig_score) in enumerate(rows, 1):
            print(f"\n  [{i}] id={row_id}")
            print(f"      significance_score : {sig_score}")
            print(f"      conversation_title : {conv_title!r}")
            print(f"      emotion            : {emotion!r}")
            print(f"      personality        : {personality!r}")
            print(f"      embedding_model    : {emb_model!r}")
            print(f"      user_input (100ch) : {user_input_preview!r}")
    else:
        print("  (no rows returned)")

    # ------------------------------------------------------------------ #
    # (e) fts_vector for 3 sample imported rows                           #
    # ------------------------------------------------------------------ #
    print(f"\n{SEP}")
    print("(e) fts_vector for 3 sample rows with conversation_id IS NOT NULL")
    print("    (first 200 chars of fts_vector to verify it is populated)")
    print(SEP)
    cur.execute("""
        SELECT
            id,
            LEFT(fts_vector::text, 200) AS fts_preview
        FROM memories
        WHERE conversation_id IS NOT NULL
          AND fts_vector IS NOT NULL
        ORDER BY significance_score DESC NULLS LAST
        LIMIT 3;
    """)
    rows = cur.fetchall()
    if rows:
        for row_id, fts_preview in rows:
            print(f"\n  id={row_id}")
            print(f"  fts_vector (200ch): {fts_preview!r}")
    else:
        # fts_vector might be a generated column that can't be IS NOT NULL filtered
        # in some Postgres versions — fall back without the NULL check
        print("  (no rows with fts_vector IS NOT NULL; retrying without NULL filter)")
        cur.execute("""
            SELECT
                id,
                LEFT(fts_vector::text, 200) AS fts_preview
            FROM memories
            WHERE conversation_id IS NOT NULL
            ORDER BY significance_score DESC NULLS LAST
            LIMIT 3;
        """)
        rows = cur.fetchall()
        if rows:
            for row_id, fts_preview in rows:
                print(f"\n  id={row_id}")
                print(f"  fts_vector (200ch): {fts_preview!r}")
        else:
            print("  (still no rows returned — table may be empty or fts_vector column missing)")

    print(f"\n{SEP}")
    print("Diagnostics complete.")
    print(SEP)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        logger.error("Diagnostic failed: %s", exc, exc_info=True)
        sys.exit(1)

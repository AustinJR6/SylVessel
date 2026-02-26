"""
Re-embed imported ChatGPT memories using OpenAI embeddings.

The original importer used sentence-transformers/all-MiniLM-L6-v2 (a local
model) to generate embeddings. The live system uses OpenAI text-embedding-3-small.
These are completely different vector spaces — even though both are 384-dimensional,
pgvector cosine distance between them is meaningless, so old memories never
surface in semantic search.

This script:
  1. Targets all memories with conversation_id set (the ChatGPT-imported ones).
  2. Truncates each text to 6000 chars before embedding so no single item
     exceeds OpenAI's 8192-token-per-item limit.
  3. Sends texts in batches of 50 per API call (efficient but safe).
  4. Gets a fresh DB connection every batch to survive long rate-limit pauses.
  5. Tags successfully re-embedded rows with embedding_model='openai-text-embedding-3-small'
     so future runs can skip them.

Usage:
  python tools/reembed_imported_memories.py [--dry-run] [--batch 50] [--force]

  --force  Re-embed even rows already tagged with the correct model.
           Default: skip already-tagged rows.
"""

import sys
import os
import time
import argparse
import importlib.util
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIM = 384
MAX_TEXT_CHARS = 6000   # ~1500 tokens — well under the 8192-token-per-item limit


def _load_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
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


def _load_supabase_client():
    spec = importlib.util.spec_from_file_location(
        "supabase_client", ROOT / "memory" / "supabase_client.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_embedding_model_column(get_connection):
    """Add embedding_model column if it doesn't exist yet."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding_model TEXT;"
        )
        conn.commit()
        logger.info("embedding_model column ready.")
    except Exception as e:
        conn.rollback()
        logger.warning(f"Could not add embedding_model column: {e}")


def embed_batch(client, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts with retry.
    Each text is already truncated before calling this function.
    """
    for attempt in range(4):
        try:
            resp = client.embeddings.create(
                model=OPENAI_MODEL, input=texts, dimensions=OPENAI_DIM
            )
            return [item.embedding for item in resp.data]
        except Exception as e:
            err = str(e)
            if "maximum context length" in err:
                # Shouldn't happen after truncation but halve texts as last resort
                texts = [t[: len(t) // 2] or "." for t in texts]
                logger.warning(f"Token limit hit on attempt {attempt+1}, halved text lengths")
                continue
            if attempt == 3:
                raise
            wait = 2 ** attempt
            logger.warning(f"Embed attempt {attempt+1} failed: {e}. Waiting {wait}s...")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Re-embed imported memories with OpenAI")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no writes")
    parser.add_argument("--batch", type=int, default=50, help="Texts per API call (default 50)")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Re-embed even already-tagged rows")
    args = parser.parse_args()

    _load_env()

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    logger.info(f"Model: {OPENAI_MODEL}  dim: {OPENAI_DIM}  batch: {args.batch}")

    from openai import OpenAI
    oai = OpenAI(api_key=openai_key)

    sc = _load_supabase_client()
    get_connection = sc.get_connection

    # Ensure the tracking column exists
    _ensure_embedding_model_column(get_connection)

    # Build query — skip already-tagged rows unless --force
    if args.force:
        where = "conversation_id IS NOT NULL"
    else:
        where = """
            conversation_id IS NOT NULL
            AND (embedding_model IS NULL OR embedding_model != 'openai-text-embedding-3-small')
        """

    conn = get_connection()
    cur = conn.cursor()
    logger.info("Querying target rows...")
    cur.execute(f"SELECT id, user_input, sylana_response FROM memories WHERE {where} ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    if args.limit > 0:
        rows = rows[: args.limit]

    total = len(rows)
    logger.info(f"Found {total} rows to re-embed.")

    if args.dry_run or total == 0:
        logger.info("Dry run — no writes performed." if args.dry_run else "Nothing to do.")
        sys.exit(0)

    updated = 0
    errors = 0
    batch_size = max(1, args.batch)

    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        ids = [r[0] for r in batch]

        # Truncate each text individually before sending
        texts = [
            (f"User: {r[1] or ''}\nSylana: {r[2] or ''}")[:MAX_TEXT_CHARS] or "."
            for r in batch
        ]

        # Embed the batch
        try:
            embeddings = embed_batch(oai, texts)
        except Exception as e:
            logger.error(f"Batch {batch_start}-{batch_start+len(batch)} embedding failed: {e}")
            errors += len(batch)
            continue

        # Get a fresh DB connection for each batch to survive rate-limit pauses
        try:
            conn = get_connection()
            cur = conn.cursor()
        except Exception as e:
            logger.error(f"DB reconnect failed at batch {batch_start}: {e}")
            errors += len(batch)
            continue

        batch_ok = 0
        for row_id, embedding in zip(ids, embeddings):
            try:
                cur.execute(
                    """
                    UPDATE memories
                    SET embedding = %s::vector,
                        personality = COALESCE(personality, 'sylana'),
                        embedding_model = 'openai-text-embedding-3-small'
                    WHERE id = %s
                    """,
                    (embedding, row_id),
                )
                batch_ok += 1
            except Exception as e:
                logger.warning(f"Row {row_id} update failed: {e}")
                errors += 1

        try:
            conn.commit()
            updated += batch_ok
        except Exception as e:
            logger.error(f"Commit failed at batch {batch_start}: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
            errors += len(batch)
            continue

        done = min(batch_start + batch_size, total)
        logger.info(f"  {done}/{total}  ({updated} updated, {errors} errors)")
        time.sleep(0.1)

    logger.info(f"\nDone. {updated} updated, {errors} errors.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

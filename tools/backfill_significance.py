"""
Backfill significance_score and feeling_weight for ChatGPT-imported memories.

The chatgpt_importer didn't compute these fields — it left them at the
column defaults (significance_score=0.5, feeling_weight=0.5). Meanwhile,
live conversations computed by store_conversation() have values of 0.9–1.5.

This scoring gap causes imported memories to lose the continuity_score ranking
against recent live conversations even when semantically more relevant.

This script re-derives significance_score and feeling_weight from the
`intensity` and `weight` fields the importer DID store, using the same
formulas as MemoryManager.

Safe to run multiple times — only updates rows where significance_score
is still at the default 0.5 and conversation_id IS NOT NULL.

Usage:
  python tools/backfill_significance.py [--dry-run] [--force]

  --force  Update all imported rows, even those already updated.
"""

import sys
import os
import argparse
import logging
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Emotion weights — must match EMOTION_WEIGHTS in memory_manager.py
EMOTION_WEIGHTS = {
    "ecstatic": 2.0,
    "devastated": 2.0,
    "happy": 1.5,
    "sad": 1.5,
    "neutral": 1.0,
}

MEMORY_TYPE_WEIGHTS = {
    "autobiographical": 1.2,
    "relational": 1.35,
    "contextual": 1.0,
    "emotional": 1.25,
}


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


def _compute_feeling_weight(emotion: str, intensity: int) -> float:
    """Mirror of MemoryManager._compute_feeling_weight."""
    base = EMOTION_WEIGHTS.get((emotion or "neutral").lower(), 1.0)
    normalized = max(0.1, min(1.0, (intensity or 5) / 10.0))
    return round(max(0.1, min(2.5, base * normalized)), 3)


def _compute_significance(
    memory_type: str, feeling_weight: float, comfort_level: float
) -> float:
    """Mirror of MemoryManager._compute_significance_score."""
    type_weight = MEMORY_TYPE_WEIGHTS.get(memory_type or "contextual", 1.0)
    score = (0.55 * feeling_weight) + (0.25 * type_weight) + (0.2 * comfort_level)
    return round(max(0.05, min(2.5, score)), 3)


def _infer_memory_type(user_input: str, sylana_response: str, emotion: str) -> str:
    """Rough classification matching MemoryManager._classify_memory_type."""
    text = f"{(user_input or '').lower()} {(sylana_response or '').lower()}"
    if any(k in text for k in ["i feel", "i'm feeling", "this felt", "energy", "comfort", "safe with you"]):
        return "emotional"
    if any(k in text for k in ["we", "us", "our", "relationship", "between us", "bond"]):
        return "relational"
    if any(k in text for k in ["i remember", "i learned", "i changed", "growth", "who i am"]):
        return "autobiographical"
    if emotion in {"devastated", "sad", "happy", "ecstatic"}:
        return "emotional"
    return "contextual"


def _comfort_level_from_weight(weight: int, emotion: str) -> float:
    """Approximate comfort_level from the importer's weight field (1–100)."""
    base = max(0.0, min(1.0, (weight or 50) / 100.0))
    if emotion in {"happy", "ecstatic"}:
        base = min(1.0, base + 0.1)
    elif emotion in {"devastated", "sad"}:
        base = max(0.0, base - 0.1)
    return round(base, 3)


def main():
    parser = argparse.ArgumentParser(
        description="Backfill significance_score for imported memories"
    )
    parser.add_argument("--dry-run", action="store_true", help="Count rows only, no writes")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Update all imported rows, not just default-scored ones",
    )
    args = parser.parse_args()

    _load_env()

    sc = _load_supabase_client()
    get_connection = sc.get_connection

    conn = get_connection()
    cur = conn.cursor()

    if args.force:
        where = "conversation_id IS NOT NULL"
    else:
        where = "conversation_id IS NOT NULL AND significance_score <= 0.5"

    logger.info("Querying imported memories to backfill...")
    cur.execute(
        f"""
        SELECT id, user_input, sylana_response, emotion, intensity, weight, memory_type
        FROM memories
        WHERE {where}
        ORDER BY id ASC
        """,
    )
    rows = cur.fetchall()
    total = len(rows)
    logger.info(f"Found {total} imported rows to update.")

    if args.dry_run or total == 0:
        if args.dry_run:
            logger.info("Dry run — no writes performed.")
        else:
            logger.info("Nothing to do.")
        return 0

    updated = 0
    errors = 0
    BATCH = 500

    for batch_start in range(0, total, BATCH):
        batch = rows[batch_start : batch_start + BATCH]
        conn = get_connection()
        cur = conn.cursor()

        batch_ok = 0
        for row in batch:
            row_id, user_input, sylana_resp, emotion, intensity, weight, memory_type = row
            emotion = (emotion or "neutral").lower()
            intensity = int(intensity or 5)
            weight = int(weight or 50)

            feeling_weight = _compute_feeling_weight(emotion, intensity)
            comfort_level = _comfort_level_from_weight(weight, emotion)

            # Prefer stored memory_type if already set; otherwise infer
            if not memory_type or memory_type == "contextual":
                memory_type = _infer_memory_type(user_input, sylana_resp, emotion)

            significance_score = _compute_significance(memory_type, feeling_weight, comfort_level)

            try:
                cur.execute(
                    """
                    UPDATE memories
                    SET significance_score = %s,
                        feeling_weight     = %s,
                        memory_type        = %s
                    WHERE id = %s
                    """,
                    (significance_score, feeling_weight, memory_type, row_id),
                )
                batch_ok += 1
            except Exception as e:
                logger.warning(f"Row {row_id} failed: {e}")
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

        done = min(batch_start + BATCH, total)
        logger.info(f"  {done}/{total} ({updated} updated, {errors} errors)")

    logger.info(f"\nDone. {updated} updated, {errors} errors.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

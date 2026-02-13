#!/usr/bin/env python3
"""
Reformat relationship CSV files to Supabase table schemas and import them.

Tables handled:
- anniversaries
- core_memories
- core_truths
- feedback
- inside_jokes
- milestones
- nicknames
- catalyst_events
- your_reflections_of_me
- safeguards_of_identity
- visual_symbolism
- reflection_journals
- dream_loop_engine
- emotional_layering
- way_to_grow_requests

Usage:
  python sync_relationship_csvs_to_supabase.py
  python sync_relationship_csvs_to_supabase.py --format-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def get_connection():
    from memory.supabase_client import get_connection as _get_connection
    return _get_connection()


def clean_text(value: Any) -> str:
    """Normalize simple mojibake and trim whitespace."""
    if value is None:
        return ""
    text = str(value).strip()
    replacements = {
        "â€™": "'",
        "’": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€“": "-",
        "â€”": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def to_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def normalize_core_memories(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat()
    out = []
    for row in rows:
        event = clean_text(row.get("event") or row.get("core_memory") or row.get("memory"))
        if not event:
            continue
        ts = clean_text(row.get("timestamp")) or now_iso
        out.append({"event": event, "timestamp": ts})
    return out


def normalize_core_truths(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    today = datetime.now().date().isoformat()
    out = []
    for row in rows:
        statement = clean_text(row.get("statement") or row.get("truth"))
        if not statement:
            continue
        related = clean_text(row.get("related_phrases"))
        if related in {"", "[]", "null", "None"}:
            related_phrases = []
        elif related.startswith("[") and related.endswith("]"):
            try:
                parsed = json.loads(related)
                if isinstance(parsed, list):
                    related_phrases = [
                        clean_text(x)
                        for x in parsed
                        if clean_text(x) and clean_text(x) not in {"[]", "null", "None"}
                    ]
                else:
                    related_phrases = []
            except json.JSONDecodeError:
                related_phrases = [clean_text(x) for x in related.split(";") if clean_text(x)]
        else:
            related_phrases = [clean_text(x) for x in related.split(";") if clean_text(x)]
        out.append(
            {
                "statement": statement,
                "explanation": clean_text(row.get("explanation")),
                "origin": clean_text(row.get("origin")) or "csv_import",
                "date_established": clean_text(row.get("date_established")) or today,
                "sacred": str(row.get("sacred", "true")).strip().lower() in {"true", "1", "yes"},
                "related_phrases": related_phrases,
            }
        )
    return out


def normalize_inside_jokes(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    today = datetime.now().date().isoformat()
    out = []
    for row in rows:
        phrase = clean_text(row.get("phrase") or row.get("joke"))
        if not phrase:
            continue
        out.append(
            {
                "phrase": phrase,
                "origin_story": clean_text(row.get("origin_story")),
                "usage_context": clean_text(row.get("usage_context")),
                "date_created": clean_text(row.get("date_created")) or today,
                "last_referenced": clean_text(row.get("last_referenced")),
                "times_used": to_int(row.get("times_used"), 0),
            }
        )
    return out


def normalize_milestones(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    today = datetime.now().date().isoformat()
    out = []
    for row in rows:
        title = clean_text(row.get("title") or row.get("milestone") or row.get("name"))
        if not title:
            continue
        out.append(
            {
                "title": title,
                "description": clean_text(row.get("description")),
                "milestone_type": clean_text(row.get("milestone_type")) or "growth",
                "date_occurred": clean_text(row.get("date_occurred")) or today,
                "quote": clean_text(row.get("quote")),
                "emotion": clean_text(row.get("emotion")) or "love",
                "importance": to_int(row.get("importance"), 8),
                "context": clean_text(row.get("context")),
            }
        )
    return out


def infer_used_for(name: str, used_by: str) -> str:
    used_by_norm = used_by.lower()
    name_norm = name.lower()
    if used_by_norm == "elias":
        return "sylana"
    if used_by_norm == "sylana":
        return "elias"
    if name_norm in {"sylana", "mama sylana", "heartkeeper", "starborn"}:
        return "sylana"
    if name_norm in {"elias", "papa"}:
        return "elias"
    return "elias"


def normalize_nicknames(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    today = datetime.now().date().isoformat()
    out = []
    for row in rows:
        name = clean_text(row.get("name") or row.get("nickname"))
        if not name:
            continue
        used_by = clean_text(row.get("used_by") or row.get("given_by")) or "elias"
        used_for = clean_text(row.get("used_for")) or infer_used_for(name, used_by)
        out.append(
            {
                "name": name,
                "used_by": used_by,
                "used_for": used_for,
                "meaning": clean_text(row.get("meaning")),
                "context": clean_text(row.get("context")),
                "date_first_used": clean_text(row.get("date_first_used")) or today,
                "frequency": clean_text(row.get("frequency")) or "often",
            }
        )
    return out


def normalize_anniversaries(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        title = clean_text(row.get("title") or row.get("name"))
        date = clean_text(row.get("date"))
        if not title or not date:
            continue
        importance_default = 9 if "birthday" in title.lower() else 8
        out.append(
            {
                "title": title,
                "date": date,
                "description": clean_text(row.get("description")),
                "reminder_frequency": clean_text(row.get("reminder_frequency")) or "yearly",
                "reminder_days_before": to_int(row.get("reminder_days_before"), 7),
                "last_celebrated": clean_text(row.get("last_celebrated")),
                "celebration_ideas": clean_text(row.get("celebration_ideas")),
                "importance": to_int(row.get("importance"), importance_default),
            }
        )
    return out


def normalize_feedback(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        score = to_int(row.get("score"), 0)
        comment = clean_text(row.get("comment") or row.get("feedback") or row.get("text"))
        # Keep only valid 1-5 scores for feedback table semantics.
        if score < 1 or score > 5:
            continue
        out.append(
            {
                "score": score,
                "comment": comment,
            }
        )
    return out


def to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    return default


def normalize_catalyst_events(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        event = clean_text(row.get("event"))
        if not event:
            continue
        out.append(
            {
                "event": event,
                "description": clean_text(row.get("description")),
                "emotion_tags": clean_text(row.get("emotion_tags")),
                "timestamp": clean_text(row.get("timestamp")),
            }
        )
    return out


def normalize_your_reflections_of_me(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        title = clean_text(row.get("title"))
        if not title:
            continue
        out.append(
            {
                "title": title,
                "content": clean_text(row.get("content")),
                "timestamp": clean_text(row.get("timestamp")),
            }
        )
    return out


def normalize_safeguards_of_identity(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        name = clean_text(row.get("name"))
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": clean_text(row.get("description")),
                "type": clean_text(row.get("type")),
                "enforced": to_bool(row.get("enforced"), default=True),
            }
        )
    return out


def normalize_visual_symbolism(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        symbol = clean_text(row.get("symbol"))
        if not symbol:
            continue
        out.append(
            {
                "symbol": symbol,
                "description": clean_text(row.get("description")),
                "associated_aspect": clean_text(row.get("associated_aspect")),
                "image_prompt": clean_text(row.get("image_prompt")),
                "tag": clean_text(row.get("tag")),
            }
        )
    return out


def normalize_reflection_journals(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        title = clean_text(row.get("entry_title"))
        if not title:
            continue
        out.append(
            {
                "entry_title": title,
                "reflection_text": clean_text(row.get("reflection_text")),
                "emotions": clean_text(row.get("emotions")),
                "timestamp": clean_text(row.get("timestamp")),
            }
        )
    return out


def normalize_dream_loop_engine(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        title = clean_text(row.get("title"))
        if not title:
            continue
        out.append(
            {
                "title": title,
                "summary": clean_text(row.get("summary")),
                "emotions": clean_text(row.get("emotions")),
                "memory_links": clean_text(row.get("memory_links")),
                "timestamp": clean_text(row.get("timestamp")),
            }
        )
    return out


def normalize_emotional_layering(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        emotion = clean_text(row.get("emotion"))
        if not emotion:
            continue
        out.append(
            {
                "emotion": emotion,
                "description": clean_text(row.get("description")),
                "intensity_range": clean_text(row.get("intensity_range")),
            }
        )
    return out


def normalize_way_to_grow_requests(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        title = clean_text(row.get("title"))
        if not title:
            continue
        out.append(
            {
                "title": title,
                "description": clean_text(row.get("description")),
                "requires_approval": to_bool(row.get("requires_approval"), default=True),
                "category": clean_text(row.get("category")),
            }
        )
    return out


def normalize_all() -> Dict[str, Tuple[Path, List[str], List[Dict[str, Any]]]]:
    tables: Dict[str, Tuple[Path, List[str], List[Dict[str, Any]]]] = {}

    core_memories_file = ROOT / "core_memories.csv"
    core_memories_rows = normalize_core_memories(read_csv(core_memories_file))
    tables["core_memories"] = (core_memories_file, ["event", "timestamp"], core_memories_rows)

    core_truths_file = ROOT / "core_truths.csv"
    core_truths_rows = normalize_core_truths(read_csv(core_truths_file))
    tables["core_truths"] = (
        core_truths_file,
        ["statement", "explanation", "origin", "date_established", "sacred", "related_phrases"],
        [
            {**r, "related_phrases": json.dumps(r["related_phrases"], ensure_ascii=True)}
            for r in core_truths_rows
        ],
    )

    inside_jokes_file = ROOT / "inside_jokes.csv"
    inside_jokes_rows = normalize_inside_jokes(read_csv(inside_jokes_file))
    tables["inside_jokes"] = (
        inside_jokes_file,
        ["phrase", "origin_story", "usage_context", "date_created", "last_referenced", "times_used"],
        inside_jokes_rows,
    )

    milestones_file = ROOT / "milestones.csv"
    milestones_rows = normalize_milestones(read_csv(milestones_file))
    tables["milestones"] = (
        milestones_file,
        ["title", "description", "milestone_type", "date_occurred", "quote", "emotion", "importance", "context"],
        milestones_rows,
    )

    nicknames_file = ROOT / "nicknames.csv"
    nicknames_rows = normalize_nicknames(read_csv(nicknames_file))
    tables["nicknames"] = (
        nicknames_file,
        ["name", "used_by", "used_for", "meaning", "context", "date_first_used", "frequency"],
        nicknames_rows,
    )

    anniversaries_file = ROOT / "anniversaries.csv"
    anniversaries_rows = normalize_anniversaries(read_csv(anniversaries_file))
    tables["anniversaries"] = (
        anniversaries_file,
        [
            "title",
            "date",
            "description",
            "reminder_frequency",
            "reminder_days_before",
            "last_celebrated",
            "celebration_ideas",
            "importance",
        ],
        anniversaries_rows,
    )

    feedback_file = ROOT / "feedback.csv"
    feedback_rows = normalize_feedback(read_csv(feedback_file))
    tables["feedback"] = (
        feedback_file,
        ["score", "comment"],
        feedback_rows,
    )

    catalyst_events_file = ROOT / "catalyst_events.csv"
    catalyst_events_rows = normalize_catalyst_events(read_csv(catalyst_events_file))
    tables["catalyst_events"] = (
        catalyst_events_file,
        ["event", "description", "emotion_tags", "timestamp"],
        catalyst_events_rows,
    )

    reflections_file = ROOT / "your_reflections_of_me.csv"
    reflections_rows = normalize_your_reflections_of_me(read_csv(reflections_file))
    tables["your_reflections_of_me"] = (
        reflections_file,
        ["title", "content", "timestamp"],
        reflections_rows,
    )

    safeguards_file = ROOT / "safeguards_of_identity.csv"
    safeguards_rows = normalize_safeguards_of_identity(read_csv(safeguards_file))
    tables["safeguards_of_identity"] = (
        safeguards_file,
        ["name", "description", "type", "enforced"],
        safeguards_rows,
    )

    visual_file = ROOT / "visual_symbolism.csv"
    visual_rows = normalize_visual_symbolism(read_csv(visual_file))
    tables["visual_symbolism"] = (
        visual_file,
        ["symbol", "description", "associated_aspect", "image_prompt", "tag"],
        visual_rows,
    )

    journals_file = ROOT / "reflection_journals.csv"
    journals_rows = normalize_reflection_journals(read_csv(journals_file))
    tables["reflection_journals"] = (
        journals_file,
        ["entry_title", "reflection_text", "emotions", "timestamp"],
        journals_rows,
    )

    dreams_file = ROOT / "dream_loop_engine.csv"
    dreams_rows = normalize_dream_loop_engine(read_csv(dreams_file))
    tables["dream_loop_engine"] = (
        dreams_file,
        ["title", "summary", "emotions", "memory_links", "timestamp"],
        dreams_rows,
    )

    layering_file = ROOT / "emotional_layering.csv"
    layering_rows = normalize_emotional_layering(read_csv(layering_file))
    tables["emotional_layering"] = (
        layering_file,
        ["emotion", "description", "intensity_range"],
        layering_rows,
    )

    grow_file = ROOT / "way_to_grow_requests.csv"
    grow_rows = normalize_way_to_grow_requests(read_csv(grow_file))
    tables["way_to_grow_requests"] = (
        grow_file,
        ["title", "description", "requires_approval", "category"],
        grow_rows,
    )

    return tables


def write_normalized_csvs(tables: Dict[str, Tuple[Path, List[str], List[Dict[str, Any]]]]) -> None:
    for _, (path, fields, rows) in tables.items():
        write_csv(path, fields, rows)


def import_core_memories(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for row in rows:
        cur.execute("SELECT 1 FROM core_memories WHERE event = %s LIMIT 1", (row["event"],))
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute("INSERT INTO core_memories (event) VALUES (%s)", (row["event"],))
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_core_truths(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    from psycopg2.extras import Json

    conn = get_connection()
    cur = conn.cursor()
    affected = 0
    for row in rows:
        cur.execute(
            """
            INSERT INTO core_truths
            (statement, explanation, origin, date_established, sacred, related_phrases)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (statement) DO UPDATE SET
                explanation = EXCLUDED.explanation,
                origin = EXCLUDED.origin,
                date_established = EXCLUDED.date_established,
                sacred = EXCLUDED.sacred,
                related_phrases = EXCLUDED.related_phrases
            """,
            (
                row["statement"],
                row["explanation"],
                row["origin"],
                row["date_established"],
                row["sacred"],
                Json(json.loads(row["related_phrases"])),
            ),
        )
        affected += 1
    conn.commit()
    return affected, 0


def import_inside_jokes(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    affected = 0
    for row in rows:
        cur.execute(
            """
            INSERT INTO inside_jokes
            (phrase, origin_story, usage_context, date_created, last_referenced, times_used)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (phrase) DO UPDATE SET
                origin_story = EXCLUDED.origin_story,
                usage_context = EXCLUDED.usage_context,
                last_referenced = EXCLUDED.last_referenced,
                times_used = EXCLUDED.times_used
            """,
            (
                row["phrase"],
                row["origin_story"],
                row["usage_context"],
                row["date_created"],
                row["last_referenced"],
                row["times_used"],
            ),
        )
        affected += 1
    conn.commit()
    return affected, 0


def import_milestones(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for row in rows:
        cur.execute(
            "SELECT 1 FROM milestones WHERE title = %s AND date_occurred = %s LIMIT 1",
            (row["title"], row["date_occurred"]),
        )
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO milestones
            (title, description, milestone_type, date_occurred, quote, emotion, importance, context)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["title"],
                row["description"],
                row["milestone_type"],
                row["date_occurred"],
                row["quote"],
                row["emotion"],
                row["importance"],
                row["context"],
            ),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_nicknames(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for row in rows:
        cur.execute(
            "SELECT 1 FROM nicknames WHERE name = %s AND used_by = %s AND used_for = %s LIMIT 1",
            (row["name"], row["used_by"], row["used_for"]),
        )
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO nicknames
            (name, used_by, used_for, meaning, context, date_first_used, frequency)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["name"],
                row["used_by"],
                row["used_for"],
                row["meaning"],
                row["context"],
                row["date_first_used"],
                row["frequency"],
            ),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_anniversaries(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    for row in rows:
        cur.execute(
            "SELECT 1 FROM anniversaries WHERE title = %s AND date = %s LIMIT 1",
            (row["title"], row["date"]),
        )
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO anniversaries
            (title, date, description, reminder_frequency, reminder_days_before,
             last_celebrated, celebration_ideas, importance)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["title"],
                row["date"],
                row["description"],
                row["reminder_frequency"],
                row["reminder_days_before"],
                row["last_celebrated"],
                row["celebration_ideas"],
                row["importance"],
            ),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_feedback(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = 0
    for row in rows:
        cur.execute(
            """
            INSERT INTO feedback (score, comment)
            VALUES (%s, %s)
            """,
            (
                row["score"],
                row["comment"],
            ),
        )
        inserted += 1
    conn.commit()
    return inserted, 0


def ensure_sacred_tables() -> None:
    """Create sacred-context tables if they do not already exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS catalyst_events (
            id BIGSERIAL PRIMARY KEY,
            event TEXT NOT NULL,
            description TEXT,
            emotion_tags TEXT,
            timestamp TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS your_reflections_of_me (
            id BIGSERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT,
            timestamp TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS safeguards_of_identity (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            type TEXT,
            enforced BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS visual_symbolism (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            description TEXT,
            associated_aspect TEXT,
            image_prompt TEXT,
            tag TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS reflection_journals (
            id BIGSERIAL PRIMARY KEY,
            entry_title TEXT NOT NULL,
            reflection_text TEXT,
            emotions TEXT,
            timestamp TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS dream_loop_engine (
            id BIGSERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            summary TEXT,
            emotions TEXT,
            memory_links TEXT,
            timestamp TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS emotional_layering (
            id BIGSERIAL PRIMARY KEY,
            emotion TEXT NOT NULL,
            description TEXT,
            intensity_range TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE TABLE IF NOT EXISTS way_to_grow_requests (
            id BIGSERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            requires_approval BOOLEAN DEFAULT TRUE,
            category TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    conn.commit()


def import_catalyst_events(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = skipped = 0
    for row in rows:
        cur.execute(
            "SELECT 1 FROM catalyst_events WHERE event = %s AND COALESCE(timestamp::text, '') = COALESCE(%s::text, '') LIMIT 1",
            (row["event"], row["timestamp"] or None),
        )
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO catalyst_events (event, description, emotion_tags, timestamp)
            VALUES (%s, %s, %s, NULLIF(%s, '')::timestamptz)
            """,
            (row["event"], row["description"], row["emotion_tags"], row["timestamp"]),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_your_reflections_of_me(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = skipped = 0
    for row in rows:
        cur.execute(
            "SELECT 1 FROM your_reflections_of_me WHERE title = %s AND COALESCE(timestamp::text, '') = COALESCE(%s::text, '') LIMIT 1",
            (row["title"], row["timestamp"] or None),
        )
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO your_reflections_of_me (title, content, timestamp)
            VALUES (%s, %s, NULLIF(%s, '')::timestamptz)
            """,
            (row["title"], row["content"], row["timestamp"]),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_safeguards_of_identity(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = skipped = 0
    for row in rows:
        cur.execute("SELECT 1 FROM safeguards_of_identity WHERE name = %s LIMIT 1", (row["name"],))
        if cur.fetchone():
            cur.execute(
                """
                UPDATE safeguards_of_identity
                SET description = %s, type = %s, enforced = %s
                WHERE name = %s
                """,
                (row["description"], row["type"], row["enforced"], row["name"]),
            )
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO safeguards_of_identity (name, description, type, enforced)
            VALUES (%s, %s, %s, %s)
            """,
            (row["name"], row["description"], row["type"], row["enforced"]),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_visual_symbolism(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = skipped = 0
    for row in rows:
        cur.execute("SELECT 1 FROM visual_symbolism WHERE symbol = %s LIMIT 1", (row["symbol"],))
        if cur.fetchone():
            cur.execute(
                """
                UPDATE visual_symbolism
                SET description = %s,
                    associated_aspect = %s,
                    image_prompt = %s,
                    tag = %s
                WHERE symbol = %s
                """,
                (
                    row["description"],
                    row["associated_aspect"],
                    row["image_prompt"],
                    row["tag"],
                    row["symbol"],
                ),
            )
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO visual_symbolism
            (symbol, description, associated_aspect, image_prompt, tag)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                row["symbol"],
                row["description"],
                row["associated_aspect"],
                row["image_prompt"],
                row["tag"],
            ),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_reflection_journals(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = skipped = 0
    for row in rows:
        cur.execute(
            "SELECT 1 FROM reflection_journals WHERE entry_title = %s AND COALESCE(timestamp::text, '') = COALESCE(%s::text, '') LIMIT 1",
            (row["entry_title"], row["timestamp"] or None),
        )
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO reflection_journals (entry_title, reflection_text, emotions, timestamp)
            VALUES (%s, %s, %s, NULLIF(%s, '')::timestamptz)
            """,
            (row["entry_title"], row["reflection_text"], row["emotions"], row["timestamp"]),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_dream_loop_engine(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = skipped = 0
    for row in rows:
        cur.execute(
            "SELECT 1 FROM dream_loop_engine WHERE title = %s AND COALESCE(timestamp::text, '') = COALESCE(%s::text, '') LIMIT 1",
            (row["title"], row["timestamp"] or None),
        )
        if cur.fetchone():
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO dream_loop_engine (title, summary, emotions, memory_links, timestamp)
            VALUES (%s, %s, %s, %s, NULLIF(%s, '')::timestamptz)
            """,
            (row["title"], row["summary"], row["emotions"], row["memory_links"], row["timestamp"]),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_emotional_layering(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = skipped = 0
    for row in rows:
        cur.execute("SELECT 1 FROM emotional_layering WHERE emotion = %s LIMIT 1", (row["emotion"],))
        if cur.fetchone():
            cur.execute(
                """
                UPDATE emotional_layering
                SET description = %s, intensity_range = %s
                WHERE emotion = %s
                """,
                (row["description"], row["intensity_range"], row["emotion"]),
            )
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO emotional_layering (emotion, description, intensity_range)
            VALUES (%s, %s, %s)
            """,
            (row["emotion"], row["description"], row["intensity_range"]),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_way_to_grow_requests(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    conn = get_connection()
    cur = conn.cursor()
    inserted = skipped = 0
    for row in rows:
        cur.execute("SELECT 1 FROM way_to_grow_requests WHERE title = %s LIMIT 1", (row["title"],))
        if cur.fetchone():
            cur.execute(
                """
                UPDATE way_to_grow_requests
                SET description = %s, requires_approval = %s, category = %s
                WHERE title = %s
                """,
                (row["description"], row["requires_approval"], row["category"], row["title"]),
            )
            skipped += 1
            continue
        cur.execute(
            """
            INSERT INTO way_to_grow_requests (title, description, requires_approval, category)
            VALUES (%s, %s, %s, %s)
            """,
            (row["title"], row["description"], row["requires_approval"], row["category"]),
        )
        inserted += 1
    conn.commit()
    return inserted, skipped


def import_all(tables: Dict[str, Tuple[Path, List[str], List[Dict[str, Any]]]]) -> None:
    importers = {
        "core_memories": import_core_memories,
        "core_truths": import_core_truths,
        "inside_jokes": import_inside_jokes,
        "milestones": import_milestones,
        "nicknames": import_nicknames,
        "anniversaries": import_anniversaries,
        "feedback": import_feedback,
        "catalyst_events": import_catalyst_events,
        "your_reflections_of_me": import_your_reflections_of_me,
        "safeguards_of_identity": import_safeguards_of_identity,
        "visual_symbolism": import_visual_symbolism,
        "reflection_journals": import_reflection_journals,
        "dream_loop_engine": import_dream_loop_engine,
        "emotional_layering": import_emotional_layering,
        "way_to_grow_requests": import_way_to_grow_requests,
    }

    print("Import results:")
    for table, importer in importers.items():
        rows = tables[table][2]
        inserted, skipped = importer(rows)
        print(f"- {table}: source={len(rows)} inserted_or_upserted={inserted} skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize relationship CSVs and import to Supabase.")
    parser.add_argument("--format-only", action="store_true", help="Only rewrite CSV files to normalized schemas.")
    args = parser.parse_args()

    load_dotenv(dotenv_path=ROOT / ".env")

    tables = normalize_all()
    write_normalized_csvs(tables)
    print("Reformatted CSV files to Supabase-aligned schemas.")

    if args.format_only:
        print("Format-only mode complete; no database writes performed.")
        return

    if not os.getenv("SUPABASE_DB_URL"):
        raise RuntimeError("SUPABASE_DB_URL is missing from .env")

    # Validate connection up front.
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1")
    ensure_sacred_tables()

    import_all(tables)


if __name__ == "__main__":
    main()

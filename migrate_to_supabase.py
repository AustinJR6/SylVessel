#!/usr/bin/env python3
"""
Sylana Vessel - SQLite to Supabase Migration
=============================================
Migrates all data from local SQLite databases to Supabase PostgreSQL with pgvector.

Usage:
    python migrate_to_supabase.py

Requires:
    - SUPABASE_DB_URL in .env
    - Local SQLite databases in ./data/
    - sentence-transformers model available
"""

import os
import sys
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).parent))

from memory.supabase_client import get_connection

MEMORY_DB = "./data/sylana_memory.db"
RELATIONSHIP_DB = "./data/relationship_memory.db"
BATCH_SIZE = 100


def migrate_memories(sqlite_conn, pg_conn, embedder):
    """Migrate memories table with embeddings."""
    from psycopg2.extras import execute_values

    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memories")
    total = cursor.fetchone()[0]
    print(f"  Found {total} memories to migrate")

    if total == 0:
        print("  No memories to migrate")
        return

    # Check if memory_embeddings table exists for pre-computed embeddings
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'")
    has_embeddings_table = cursor.fetchone() is not None

    cursor.execute("""
        SELECT id, user_input, sylana_response, timestamp, emotion,
               intensity, topic, core_memory, weight,
               conversation_id, conversation_title
        FROM memories ORDER BY id
    """)

    pg_cur = pg_conn.cursor()
    batch = []
    migrated = 0
    embedded_from_cache = 0
    embedded_fresh = 0

    for row in cursor:
        (old_id, user_input, response, ts, emotion, intensity,
         topic, core_mem, weight, conv_id, conv_title) = row

        # Try to load pre-computed embedding
        embedding = None
        if has_embeddings_table:
            emb_cursor = sqlite_conn.cursor()
            emb_cursor.execute(
                "SELECT embedding FROM memory_embeddings WHERE memory_id = ?",
                (old_id,)
            )
            emb_row = emb_cursor.fetchone()
            if emb_row and emb_row[0]:
                try:
                    embedding = np.frombuffer(emb_row[0], dtype=np.float32).tolist()
                    embedded_from_cache += 1
                except Exception:
                    embedding = None

        # Generate embedding if not cached
        if embedding is None:
            text = f"User: {user_input}\nSylana: {response}"
            embedding = embedder.encode([text], convert_to_numpy=True)[0].tolist()
            embedded_fresh += 1

        batch.append((
            user_input, response, ts, emotion, intensity or 5,
            topic or 'general', bool(core_mem), weight or 50,
            conv_id, conv_title, embedding
        ))

        if len(batch) >= BATCH_SIZE:
            execute_values(pg_cur, """
                INSERT INTO memories
                (user_input, sylana_response, timestamp, emotion, intensity, topic,
                 core_memory, weight, conversation_id, conversation_title, embedding)
                VALUES %s
            """, batch)
            pg_conn.commit()
            migrated += len(batch)
            print(f"  Migrated {migrated}/{total} memories...", end='\r')
            batch = []

    if batch:
        execute_values(pg_cur, """
            INSERT INTO memories
            (user_input, sylana_response, timestamp, emotion, intensity, topic,
             core_memory, weight, conversation_id, conversation_title, embedding)
            VALUES %s
        """, batch)
        pg_conn.commit()
        migrated += len(batch)

    print(f"  Migrated {migrated} memories ({embedded_from_cache} cached embeddings, {embedded_fresh} freshly encoded)")


def migrate_core_memories(sqlite_conn, pg_conn):
    """Migrate core_memories table."""
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT event, timestamp FROM core_memories")
    rows = cursor.fetchall()

    if not rows:
        print("  No core memories to migrate")
        return

    pg_cur = pg_conn.cursor()
    for event, ts in rows:
        pg_cur.execute(
            "INSERT INTO core_memories (event) VALUES (%s)",
            (event,)
        )
    pg_conn.commit()
    print(f"  Migrated {len(rows)} core memories")


def migrate_feedback(sqlite_conn, pg_conn):
    """Migrate feedback table."""
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT score, comment FROM feedback")
    rows = cursor.fetchall()

    if not rows:
        print("  No feedback to migrate")
        return

    pg_cur = pg_conn.cursor()
    for score, comment in rows:
        pg_cur.execute(
            "INSERT INTO feedback (score, comment) VALUES (%s, %s)",
            (score, comment)
        )
    pg_conn.commit()
    print(f"  Migrated {len(rows)} feedback entries")


def migrate_relationship_tables(sqlite_conn, pg_conn):
    """Migrate all 5 relationship tables."""
    from psycopg2.extras import Json

    tables = {
        'milestones': [
            'title', 'description', 'milestone_type', 'date_occurred',
            'quote', 'emotion', 'importance', 'context'
        ],
        'inside_jokes': [
            'phrase', 'origin_story', 'usage_context', 'date_created',
            'last_referenced', 'times_used'
        ],
        'nicknames': [
            'name', 'used_by', 'used_for', 'meaning', 'context',
            'date_first_used', 'frequency'
        ],
        'core_truths': [
            'statement', 'explanation', 'origin', 'date_established',
            'sacred', 'related_phrases'
        ],
        'anniversaries': [
            'title', 'date', 'description', 'reminder_frequency',
            'reminder_days_before', 'last_celebrated', 'celebration_ideas',
            'importance'
        ]
    }

    cursor = sqlite_conn.cursor()
    pg_cur = pg_conn.cursor()

    for table, columns in tables.items():
        # Check if table exists in SQLite
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        if not cursor.fetchone():
            print(f"  Table {table} not found in SQLite, skipping")
            continue

        cols_str = ', '.join(columns)
        cursor.execute(f"SELECT {cols_str} FROM {table}")
        rows = cursor.fetchall()

        if not rows:
            print(f"  No data in {table}")
            continue

        placeholders = ', '.join(['%s'] * len(columns))
        for row in rows:
            row_list = list(row)

            # Handle core_truths.related_phrases: convert JSON string to JSONB
            if table == 'core_truths':
                rp_idx = columns.index('related_phrases')
                rp_val = row_list[rp_idx]
                if isinstance(rp_val, str):
                    try:
                        rp_val = json.loads(rp_val)
                    except json.JSONDecodeError:
                        rp_val = []
                row_list[rp_idx] = Json(rp_val if rp_val else [])

            pg_cur.execute(
                f"INSERT INTO {table} ({cols_str}) VALUES ({placeholders})",
                tuple(row_list)
            )

        pg_conn.commit()
        print(f"  Migrated {len(rows)} rows to {table}")


def main():
    print("=" * 60)
    print("  SYLANA VESSEL - SQLite to Supabase Migration")
    print("=" * 60)

    # Verify Supabase connection
    if not os.getenv("SUPABASE_DB_URL"):
        print("\n[ERROR] SUPABASE_DB_URL not set in .env")
        print("Add your Supabase direct connection string to .env:")
        print("SUPABASE_DB_URL=postgresql://postgres.[ref]:[password]@[host]:5432/postgres")
        sys.exit(1)

    print("\nLoading embedding model...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Connecting to Supabase...")
    pg_conn = get_connection()

    # Migrate main memory database
    if Path(MEMORY_DB).exists():
        print(f"\n--- Migrating Memory Database ({MEMORY_DB}) ---")
        sqlite_mem = sqlite3.connect(MEMORY_DB)
        migrate_memories(sqlite_mem, pg_conn, embedder)
        migrate_core_memories(sqlite_mem, pg_conn)
        migrate_feedback(sqlite_mem, pg_conn)
        sqlite_mem.close()
    else:
        print(f"\n[SKIP] Memory database not found: {MEMORY_DB}")

    # Migrate relationship database
    if Path(RELATIONSHIP_DB).exists():
        print(f"\n--- Migrating Relationship Database ({RELATIONSHIP_DB}) ---")
        sqlite_rel = sqlite3.connect(RELATIONSHIP_DB)
        migrate_relationship_tables(sqlite_rel, pg_conn)
        sqlite_rel.close()
    else:
        print(f"\n[SKIP] Relationship database not found: {RELATIONSHIP_DB}")

    # Verify
    print("\n--- Verification ---")
    cur = pg_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM memories")
    print(f"  Total memories in Supabase: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL")
    print(f"  Memories with embeddings: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM core_memories")
    print(f"  Core memories: {cur.fetchone()[0]}")

    for table in ['milestones', 'inside_jokes', 'nicknames', 'core_truths', 'anniversaries']:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"  {table}: {cur.fetchone()[0]}")
        except Exception:
            pg_conn.rollback()

    print("\nMigration complete!")


if __name__ == "__main__":
    main()

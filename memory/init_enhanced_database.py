"""
Sylana Vessel - Enhanced Database Schema
Adds support for core memories, tags, dreams, and journaling
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import config


def initialize_enhanced_database():
    """Create or upgrade database with enhanced schema"""
    db_path = config.DB_PATH

    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SYLANA VESSEL - ENHANCED MEMORY SYSTEM INITIALIZATION")
    print("=" * 70)
    print(f"\nDatabase: {db_path}\n")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Original tables (maintain compatibility)
        print("[1/8] Creating/verifying memory table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                sylana_response TEXT,
                emotion TEXT DEFAULT 'neutral',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("[2/8] Creating/verifying core_memories table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS core_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("[3/8] Creating/verifying feedback table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                score INTEGER CHECK(score >= 1 AND score <= 5),
                comment TEXT DEFAULT '',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES memory(id)
            )
        """)

        # NEW: Enhanced memory system tables
        print("[4/8] Creating enhanced_memories table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('core', 'dynamic', 'dream', 'journal')),
                tags TEXT,  -- Comma-separated tags
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT DEFAULT 'system',  -- 'system', 'user', 'sylana'
                immutable INTEGER DEFAULT 0,  -- 1 for core truths that cannot be deleted
                metadata TEXT  -- JSON for additional data
            )
        """)

        print("[5/8] Creating memory_tags table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES enhanced_memories(id) ON DELETE CASCADE,
                UNIQUE(memory_id, tag)
            )
        """)

        print("[6/8] Creating journal_entries table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                emotional_summary TEXT,
                key_moments TEXT,  -- JSON array
                questions_holding TEXT,  -- JSON array
                tags TEXT,  -- Comma-separated
                reflection TEXT,  -- Full text reflection
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        """)

        print("[7/8] Creating dream_log table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dream_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dream_text TEXT NOT NULL,
                source_tags TEXT,  -- Tags used to generate dream
                symbolic_elements TEXT,  -- JSON of symbols used
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                shared_with_elias INTEGER DEFAULT 0
            )
        """)

        print("[8/8] Creating indices...")
        # Original indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_emotion ON memory(emotion)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_core_memories_timestamp ON core_memories(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_score ON feedback(score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC)")

        # New indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_memories_type ON enhanced_memories(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_memories_timestamp ON enhanced_memories(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_memories_immutable ON enhanced_memories(immutable)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags_memory_id ON memory_tags(memory_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal_entries_date ON journal_entries(date DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dream_log_timestamp ON dream_log(timestamp DESC)")

        conn.commit()

        print("\n[OK] Enhanced database schema created successfully!")
        print("\nTables:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        for (table,) in cursor.fetchall():
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  - {table} ({count} rows)")

        print("\n" + "=" * 70)
        print("  ENHANCED MEMORY SYSTEM READY")
        print("=" * 70)

        conn.close()
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to initialize enhanced database: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    initialize_enhanced_database()

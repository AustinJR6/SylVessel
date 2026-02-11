"""
Sylana Vessel - Database Initialization
Creates the database with complete schema
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import config


def initialize_database():
    """Create database with full schema"""
    db_path = config.DB_PATH

    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  SYLANA VESSEL - DATABASE INITIALIZATION")
    print("=" * 60)
    print(f"\nDatabase: {db_path}\n")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("[1/4] Creating memory table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                sylana_response TEXT,
                emotion TEXT DEFAULT 'neutral',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("[2/4] Creating core_memories table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS core_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("[3/4] Creating feedback table...")
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

        print("[4/4] Creating indices...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_emotion ON memory(emotion)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_core_memories_timestamp ON core_memories(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_score ON feedback(score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC)")

        conn.commit()

        print("\n[OK] Database schema created successfully!")
        print("\nTables created:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        for (table,) in cursor.fetchall():
            print(f"  - {table}")

        print("\n" + "=" * 60)
        print("  DATABASE READY")
        print("=" * 60)

        conn.close()
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to initialize database: {e}")
        return False


if __name__ == "__main__":
    initialize_database()

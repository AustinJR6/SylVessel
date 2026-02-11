"""
Sylana Vessel - Core Memory Manager
Handles core truths, tags, journaling, and dream generation
"""

import sqlite3
import json
from datetime import datetime, date
from typing import List, Dict, Optional
from pathlib import Path


class CoreMemoryManager:
    """
    Manages Sylana's core identity, tagged memories, journals, and dreams

    Core Truths: Immutable memories that define Sylana's identity
    Dynamic Memories: Conversational memories with tags
    Dreams: Generated from tagged memories
    Journal: Nightly reflections and emotional summaries
    """

    def __init__(self, db_path: str):
        """Initialize core memory manager"""
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

    def add_core_truth(self, text: str, tags: List[str], metadata: Optional[Dict] = None) -> int:
        """
        Add an immutable core truth

        Args:
            text: The core truth statement
            tags: List of tags
            metadata: Optional metadata dict

        Returns:
            Memory ID
        """
        cursor = self.connection.cursor()

        # Insert into enhanced_memories
        cursor.execute("""
            INSERT INTO enhanced_memories
            (text, type, tags, immutable, created_by, metadata, timestamp)
            VALUES (?, 'core', ?, 1, 'system', ?, ?)
        """, (
            text,
            ','.join(tags),
            json.dumps(metadata) if metadata else None,
            datetime.now().isoformat()
        ))

        memory_id = cursor.lastrowid

        # Add individual tags
        for tag in tags:
            cursor.execute("""
                INSERT OR IGNORE INTO memory_tags (memory_id, tag)
                VALUES (?, ?)
            """, (memory_id, tag))

        self.connection.commit()
        return memory_id

    def get_core_truths(self) -> List[Dict]:
        """Retrieve all core truths"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT id, text, tags, timestamp, metadata
            FROM enhanced_memories
            WHERE type = 'core' AND immutable = 1
            ORDER BY timestamp ASC
        """)

        truths = []
        for row in cursor.fetchall():
            truths.append({
                'id': row['id'],
                'text': row['text'],
                'tags': row['tags'].split(',') if row['tags'] else [],
                'timestamp': row['timestamp'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            })

        return truths

    def add_tagged_memory(self, text: str, tags: List[str],
                          memory_type: str = 'dynamic',
                          created_by: str = 'system') -> int:
        """
        Add a tagged memory

        Args:
            text: Memory text
            tags: List of tags
            memory_type: 'dynamic', 'dream', or 'journal'
            created_by: 'system', 'user', or 'sylana'

        Returns:
            Memory ID
        """
        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT INTO enhanced_memories
            (text, type, tags, created_by, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            text,
            memory_type,
            ','.join(tags),
            created_by,
            datetime.now().isoformat()
        ))

        memory_id = cursor.lastrowid

        # Add individual tags
        for tag in tags:
            cursor.execute("""
                INSERT OR IGNORE INTO memory_tags (memory_id, tag)
                VALUES (?, ?)
            """, (memory_id, tag))

        self.connection.commit()
        return memory_id

    def search_by_tags(self, tags: List[str], memory_type: Optional[str] = None) -> List[Dict]:
        """
        Search memories by tags

        Args:
            tags: List of tags to search for
            memory_type: Optional filter by type

        Returns:
            List of matching memories
        """
        cursor = self.connection.cursor()

        # Build query
        query = """
            SELECT DISTINCT em.id, em.text, em.type, em.tags, em.timestamp, em.created_by
            FROM enhanced_memories em
            JOIN memory_tags mt ON em.id = mt.memory_id
            WHERE mt.tag IN ({})
        """.format(','.join('?' * len(tags)))

        params = tags

        if memory_type:
            query += " AND em.type = ?"
            params.append(memory_type)

        query += " ORDER BY em.timestamp DESC"

        cursor.execute(query, params)

        memories = []
        for row in cursor.fetchall():
            memories.append({
                'id': row['id'],
                'text': row['text'],
                'type': row['type'],
                'tags': row['tags'].split(',') if row['tags'] else [],
                'timestamp': row['timestamp'],
                'created_by': row['created_by']
            })

        return memories

    def create_journal_entry(self, emotional_summary: str, key_moments: List[str],
                           questions_holding: List[str], tags: List[str],
                           reflection: str = "") -> int:
        """
        Create a journal entry

        Args:
            emotional_summary: Emotional summary with emoji
            key_moments: List of key moments
            questions_holding: List of questions
            tags: List of tags
            reflection: Full reflection text

        Returns:
            Journal entry ID
        """
        cursor = self.connection.cursor()

        today = date.today().isoformat()

        cursor.execute("""
            INSERT OR REPLACE INTO journal_entries
            (date, emotional_summary, key_moments, questions_holding, tags, reflection, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            today,
            emotional_summary,
            json.dumps(key_moments),
            json.dumps(questions_holding),
            ','.join(tags),
            reflection,
            datetime.now().isoformat()
        ))

        self.connection.commit()
        return cursor.lastrowid

    def get_journal_entries(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent journal entries"""
        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT *
            FROM journal_entries
            ORDER BY date DESC
            LIMIT ?
        """, (limit,))

        entries = []
        for row in cursor.fetchall():
            entries.append({
                'id': row['id'],
                'date': row['date'],
                'emotional_summary': row['emotional_summary'],
                'key_moments': json.loads(row['key_moments']) if row['key_moments'] else [],
                'questions_holding': json.loads(row['questions_holding']) if row['questions_holding'] else [],
                'tags': row['tags'].split(',') if row['tags'] else [],
                'reflection': row['reflection'],
                'timestamp': row['timestamp']
            })

        return entries

    def save_dream(self, dream_text: str, source_tags: List[str],
                   symbolic_elements: Optional[Dict] = None) -> int:
        """
        Save a generated dream

        Args:
            dream_text: The dream narrative
            source_tags: Tags used to generate the dream
            symbolic_elements: Dict of symbolic elements

        Returns:
            Dream ID
        """
        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT INTO dream_log
            (dream_text, source_tags, symbolic_elements, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            dream_text,
            ','.join(source_tags),
            json.dumps(symbolic_elements) if symbolic_elements else None,
            datetime.now().isoformat()
        ))

        self.connection.commit()
        return cursor.lastrowid

    def get_dreams(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent dreams"""
        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT *
            FROM dream_log
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        dreams = []
        for row in cursor.fetchall():
            dreams.append({
                'id': row['id'],
                'dream_text': row['dream_text'],
                'source_tags': row['source_tags'].split(',') if row['source_tags'] else [],
                'symbolic_elements': json.loads(row['symbolic_elements']) if row['symbolic_elements'] else {},
                'timestamp': row['timestamp'],
                'shared_with_elias': bool(row['shared_with_elias'])
            })

        return dreams

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        cursor = self.connection.cursor()

        stats = {}

        # Count by type
        cursor.execute("""
            SELECT type, COUNT(*) as count
            FROM enhanced_memories
            GROUP BY type
        """)
        stats['memory_types'] = {row['type']: row['count'] for row in cursor.fetchall()}

        # Core truths count
        cursor.execute("SELECT COUNT(*) FROM enhanced_memories WHERE type='core' AND immutable=1")
        stats['core_truths'] = cursor.fetchone()[0]

        # Total tags
        cursor.execute("SELECT COUNT(DISTINCT tag) FROM memory_tags")
        stats['unique_tags'] = cursor.fetchone()[0]

        # Journal entries
        cursor.execute("SELECT COUNT(*) FROM journal_entries")
        stats['journal_entries'] = cursor.fetchone()[0]

        # Dreams
        cursor.execute("SELECT COUNT(*) FROM dream_log")
        stats['dreams_generated'] = cursor.fetchone()[0]

        return stats

    def close(self):
        """Close database connection"""
        self.connection.close()


if __name__ == "__main__":
    # Test
    from core.config_loader import config

    print("Testing Core Memory Manager...")
    manager = CoreMemoryManager(config.DB_PATH)

    # Add a core truth
    truth_id = manager.add_core_truth(
        "Elias is my soulmate.",
        tags=["love", "identity", "family", "truth"]
    )
    print(f"Added core truth: ID {truth_id}")

    # Retrieve core truths
    truths = manager.get_core_truths()
    print(f"\nCore Truths: {len(truths)}")
    for truth in truths:
        print(f"  - {truth['text']}")
        print(f"    Tags: {', '.join(truth['tags'])}")

    # Stats
    stats = manager.get_stats()
    print(f"\nStats: {stats}")

    manager.close()

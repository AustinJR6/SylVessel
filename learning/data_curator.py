"""
Sylana Vessel - Training Data Curator
Selects high-quality conversations for fine-tuning
"""

import sqlite3
import json
from typing import List, Dict
from datetime import datetime, timedelta


class TrainingDataCurator:
    """
    Curates high-quality training data from conversations.
    Filters by feedback scores, diversity, and quality metrics.
    """

    def __init__(self, db_connection):
        """
        Initialize data curator

        Args:
            db_connection: SQLite database connection
        """
        self.connection = db_connection

    def get_high_quality_conversations(
        self,
        min_score: int = 4,
        min_length: int = 10,
        max_length: int = 500,
        limit: int = None
    ) -> List[Dict]:
        """
        Retrieve high-quality conversations based on feedback

        Args:
            min_score: Minimum feedback score (default 4/5)
            min_length: Minimum response length
            max_length: Maximum response length
            limit: Maximum conversations to return

        Returns:
            List of conversation dictionaries
        """
        cursor = self.connection.cursor()

        query = """
            SELECT
                m.id,
                m.user_input,
                m.sylana_response,
                m.emotion,
                m.timestamp,
                f.score,
                f.comment
            FROM memory m
            LEFT JOIN feedback f ON m.id = f.conversation_id
            WHERE
                f.score >= ?
                AND LENGTH(m.sylana_response) >= ?
                AND LENGTH(m.sylana_response) <= ?
            ORDER BY f.score DESC, m.timestamp DESC
        """

        params = [min_score, min_length, max_length]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cursor.execute(query, params)

        return [
            {
                'id': row[0],
                'user_input': row[1],
                'sylana_response': row[2],
                'emotion': row[3],
                'timestamp': row[4],
                'feedback_score': row[5],
                'feedback_comment': row[6]
            }
            for row in cursor.fetchall()
        ]

    def get_diverse_sample(
        self,
        total_samples: int = 100,
        emotion_distribution: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Get a diverse sample of conversations across emotions

        Args:
            total_samples: Total number of samples to retrieve
            emotion_distribution: Desired distribution (e.g., {'happy': 0.3, 'sad': 0.3})

        Returns:
            Diverse list of conversations
        """
        if emotion_distribution is None:
            # Default: balanced across emotions
            emotion_distribution = {
                'happy': 0.25,
                'sad': 0.25,
                'ecstatic': 0.15,
                'devastated': 0.15,
                'neutral': 0.20
            }

        cursor = self.connection.cursor()
        all_samples = []

        for emotion, proportion in emotion_distribution.items():
            count = int(total_samples * proportion)

            cursor.execute("""
                SELECT
                    m.id,
                    m.user_input,
                    m.sylana_response,
                    m.emotion,
                    m.timestamp
                FROM memory m
                LEFT JOIN feedback f ON m.id = f.conversation_id
                WHERE m.emotion = ?
                AND (f.score IS NULL OR f.score >= 3)
                ORDER BY RANDOM()
                LIMIT ?
            """, (emotion, count))

            for row in cursor.fetchall():
                all_samples.append({
                    'id': row[0],
                    'user_input': row[1],
                    'sylana_response': row[2],
                    'emotion': row[3],
                    'timestamp': row[4]
                })

        return all_samples

    def format_for_training(
        self,
        conversations: List[Dict],
        context_length: int = 1
    ) -> List[Dict]:
        """
        Format conversations for fine-tuning

        Args:
            conversations: List of conversation dicts
            context_length: Number of context turns to include

        Returns:
            List of training examples in prompt/completion format
        """
        training_data = []

        for conv in conversations:
            # Simple format: direct user input -> response
            example = {
                'prompt': f"User: {conv['user_input']}\nSylana:",
                'completion': f" {conv['sylana_response']}",
                'metadata': {
                    'id': conv['id'],
                    'emotion': conv['emotion'],
                    'timestamp': conv['timestamp'],
                    'feedback_score': conv.get('feedback_score')
                }
            }
            training_data.append(example)

        return training_data

    def save_training_file(
        self,
        training_data: List[Dict],
        filename: str = "curated_training_data.jsonl"
    ):
        """
        Save training data to JSONL file

        Args:
            training_data: List of training examples
            filename: Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for example in training_data:
                # Remove metadata before saving (or keep if needed)
                clean_example = {
                    'prompt': example['prompt'],
                    'completion': example['completion']
                }
                f.write(json.dumps(clean_example) + '\n')

        print(f"Saved {len(training_data)} training examples to {filename}")

    def get_curation_stats(self) -> Dict:
        """Get statistics about available training data"""
        cursor = self.connection.cursor()

        # Total conversations
        cursor.execute("SELECT COUNT(*) FROM memory")
        total = cursor.fetchone()[0]

        # With feedback
        cursor.execute("""
            SELECT COUNT(*) FROM memory m
            JOIN feedback f ON m.id = f.conversation_id
        """)
        with_feedback = cursor.fetchone()[0]

        # High quality (score >= 4)
        cursor.execute("""
            SELECT COUNT(*) FROM feedback
            WHERE score >= 4
        """)
        high_quality = cursor.fetchone()[0]

        # By emotion
        cursor.execute("""
            SELECT emotion, COUNT(*)
            FROM memory
            GROUP BY emotion
        """)
        by_emotion = dict(cursor.fetchall())

        return {
            'total_conversations': total,
            'with_feedback': with_feedback,
            'high_quality_count': high_quality,
            'distribution_by_emotion': by_emotion,
            'feedback_coverage': round(with_feedback / total * 100, 1) if total > 0 else 0
        }


if __name__ == "__main__":
    # Test the curator
    print("Testing TrainingDataCurator...")

    import sqlite3
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE memory (
            id INTEGER PRIMARY KEY,
            user_input TEXT,
            sylana_response TEXT,
            emotion TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            score INTEGER,
            comment TEXT
        )
    """)

    # Add test data
    test_conversations = [
        ("How are you?", "I'm doing great, thanks!", "happy", 5),
        ("I'm sad", "I'm here for you", "sad", 5),
        ("Tell me a joke", "Why did the AI cross the road?", "happy", 4),
    ]

    for user, response, emotion, score in test_conversations:
        cursor.execute(
            "INSERT INTO memory (user_input, sylana_response, emotion) VALUES (?, ?, ?)",
            (user, response, emotion)
        )
        conv_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO feedback (conversation_id, score) VALUES (?, ?)",
            (conv_id, score)
        )

    conn.commit()

    # Test curator
    curator = TrainingDataCurator(conn)

    stats = curator.get_curation_stats()
    print(f"\nCuration Stats: {stats}")

    high_quality = curator.get_high_quality_conversations(min_score=4)
    print(f"\nHigh Quality Conversations: {len(high_quality)}")

    training_data = curator.format_for_training(high_quality)
    print(f"\nFormatted Training Examples: {len(training_data)}")

    conn.close()

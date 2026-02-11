"""
Sylana Vessel - Feedback Collection System
Interactive feedback collection during conversations
"""

import sqlite3
from typing import Optional
from datetime import datetime


class FeedbackCollector:
    """
    Collects user feedback on conversation quality.
    Enables continuous improvement through rating system.
    """

    def __init__(self, db_connection):
        """
        Initialize feedback collector

        Args:
            db_connection: SQLite database connection
        """
        self.connection = db_connection

    def prompt_for_feedback(self, conversation_id: int, user_input: str, response: str) -> Optional[int]:
        """
        Interactively prompt user for feedback

        Args:
            conversation_id: ID of the conversation turn
            user_input: User's message
            response: Sylana's response

        Returns:
            Feedback score (1-5) or None if skipped
        """
        print("\n" + "-" * 60)
        print("💭 Quick Feedback (optional - press Enter to skip)")
        print("-" * 60)
        print(f"Your message: {user_input[:50]}...")
        print(f"Sylana said: {response[:50]}...")
        print()
        print("How was this response?")
        print("  5 ⭐⭐⭐⭐⭐ - Excellent")
        print("  4 ⭐⭐⭐⭐   - Good")
        print("  3 ⭐⭐⭐     - Okay")
        print("  2 ⭐⭐       - Poor")
        print("  1 ⭐         - Bad")
        print()

        try:
            rating_input = input("Rating (1-5 or Enter to skip): ").strip()

            if not rating_input:
                return None

            rating = int(rating_input)

            if rating < 1 or rating > 5:
                print("Invalid rating. Skipping feedback.")
                return None

            # Optional comment
            comment = input("Comment (optional): ").strip()

            # Store feedback
            self.store_feedback(conversation_id, rating, comment)

            print(f"✓ Thanks! Feedback recorded: {rating}/5")
            return rating

        except ValueError:
            print("Invalid input. Skipping feedback.")
            return None
        except KeyboardInterrupt:
            print("\nFeedback skipped.")
            return None

    def store_feedback(self, conversation_id: int, score: int, comment: str = ""):
        """Store feedback in database"""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO feedback (conversation_id, score, comment)
            VALUES (?, ?, ?)
        """, (conversation_id, score, comment))
        self.connection.commit()

    def get_feedback_stats(self):
        """Get feedback statistics"""
        cursor = self.connection.cursor()

        cursor.execute("SELECT COUNT(*), AVG(score) FROM feedback")
        count, avg_score = cursor.fetchone()

        cursor.execute("""
            SELECT score, COUNT(*)
            FROM feedback
            GROUP BY score
            ORDER BY score DESC
        """)
        distribution = dict(cursor.fetchall())

        return {
            'total': count or 0,
            'average': round(avg_score, 2) if avg_score else 0.0,
            'distribution': distribution
        }

    def should_prompt_feedback(self, turn_number: int, prompt_frequency: int = 5) -> bool:
        """
        Determine if we should prompt for feedback

        Args:
            turn_number: Current conversation turn
            prompt_frequency: Prompt every N turns

        Returns:
            True if should prompt
        """
        # Prompt every N turns, but not on first turn
        return turn_number > 0 and turn_number % prompt_frequency == 0


if __name__ == "__main__":
    # Test the feedback collector
    print("Testing FeedbackCollector...")

    import sqlite3
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create feedback table
    cursor.execute("""
        CREATE TABLE feedback (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            score INTEGER,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    collector = FeedbackCollector(conn)

    # Simulate feedback
    collector.store_feedback(1, 5, "Great response!")
    collector.store_feedback(2, 4, "Good")
    collector.store_feedback(3, 5, "Excellent!")

    stats = collector.get_feedback_stats()
    print(f"\nFeedback Stats: {stats}")

    conn.close()

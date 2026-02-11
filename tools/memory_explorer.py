"""
Sylana Vessel - Memory Explorer Tool
Visualize and explore conversation memory
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory.memory_manager import MemoryManager
from core.config_loader import config
from datetime import datetime
import sqlite3


class MemoryExplorer:
    """Interactive tool to explore Sylana's memory"""

    def __init__(self):
        self.memory = MemoryManager(config.DB_PATH)

    def show_stats(self):
        """Display comprehensive memory statistics"""
        stats = self.memory.get_stats()

        print("\n" + "=" * 70)
        print("  SYLANA MEMORY STATISTICS")
        print("=" * 70)
        print()

        print(f"Total Conversations: {stats['total_conversations']}")
        print(f"Core Memories: {stats['total_core_memories']}")
        print(f"Feedback Entries: {stats['total_feedback']}")
        print(f"Average Feedback: {stats['avg_feedback_score']}/5.0")
        print()

        print("FAISS Semantic Engine:")
        print(f"  Indexed Memories: {stats['semantic_engine']['total_memories']}")
        print(f"  Embedding Model: {stats['semantic_engine']['model']}")
        print(f"  Dimension: {stats['semantic_engine']['dimension']}")
        print()

        # Emotion distribution
        cursor = self.memory.connection.cursor()
        cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM memory
            GROUP BY emotion
            ORDER BY count DESC
        """)

        print("Emotion Distribution:")
        for emotion, count in cursor.fetchall():
            percentage = (count / stats['total_conversations'] * 100) if stats['total_conversations'] > 0 else 0
            print(f"  {emotion:12s}: {count:4d} ({percentage:.1f}%)")

        print()

    def show_recent_conversations(self, limit=10):
        """Display recent conversations"""
        history = self.memory.get_conversation_history(limit=limit)

        print("\n" + "=" * 70)
        print(f"  RECENT CONVERSATIONS (Last {limit})")
        print("=" * 70)
        print()

        for i, conv in enumerate(reversed(history), 1):
            print(f"[{i}] {conv['timestamp']} - Emotion: {conv['emotion']}")
            print(f"    You: {conv['user_input'][:60]}...")
            print(f"    Sylana: {conv['sylana_response'][:60]}...")
            print()

    def search_memories(self, query):
        """Search memories semantically"""
        print(f"\nSearching for: '{query}'")
        print("=" * 70)

        results = self.memory.recall_relevant(query, k=5, use_recency_boost=False)

        if not results['conversations']:
            print("No relevant memories found.")
            return

        print(f"\nFound {len(results['conversations'])} relevant conversations:\n")

        for i, mem in enumerate(results['conversations'], 1):
            print(f"[{i}] Similarity: {mem['similarity']:.1%} - {mem['timestamp']}")
            print(f"    Emotion: {mem['emotion']}")
            print(f"    You: {mem['user_input']}")
            print(f"    Sylana: {mem['sylana_response'][:100]}...")
            print()

        if results['core_memories']:
            print(f"Relevant Core Memories:")
            for mem in results['core_memories']:
                print(f"  • {mem['event']}")
            print()

    def show_core_memories(self):
        """Display all core memories"""
        cursor = self.memory.connection.cursor()
        cursor.execute("SELECT event, timestamp FROM core_memories ORDER BY timestamp DESC")
        core_memories = cursor.fetchall()

        print("\n" + "=" * 70)
        print("  CORE MEMORIES")
        print("=" * 70)
        print()

        if not core_memories:
            print("No core memories stored yet.")
            return

        for i, (event, timestamp) in enumerate(core_memories, 1):
            print(f"[{i}] {timestamp}")
            print(f"    {event}")
            print()

    def show_feedback_summary(self):
        """Display feedback analysis"""
        cursor = self.memory.connection.cursor()

        cursor.execute("""
            SELECT
                m.user_input,
                m.sylana_response,
                f.score,
                f.comment,
                f.timestamp
            FROM feedback f
            JOIN memory m ON f.conversation_id = m.id
            ORDER BY f.timestamp DESC
            LIMIT 10
        """)

        print("\n" + "=" * 70)
        print("  RECENT FEEDBACK")
        print("=" * 70)
        print()

        results = cursor.fetchall()

        if not results:
            print("No feedback collected yet.")
            return

        for user_input, response, score, comment, timestamp in results:
            print(f"{timestamp} - Score: {score}/5")
            print(f"  You: {user_input[:60]}...")
            print(f"  Sylana: {response[:60]}...")
            if comment:
                print(f"  Comment: {comment}")
            print()

    def interactive_menu(self):
        """Interactive exploration menu"""
        while True:
            print("\n" + "=" * 70)
            print("  MEMORY EXPLORER - Main Menu")
            print("=" * 70)
            print()
            print("  1. Show Statistics")
            print("  2. Recent Conversations")
            print("  3. Search Memories")
            print("  4. Core Memories")
            print("  5. Feedback Summary")
            print("  6. Exit")
            print()

            choice = input("Your choice (1-6): ").strip()

            if choice == "1":
                self.show_stats()
            elif choice == "2":
                limit = input("How many recent conversations? (default 10): ").strip()
                limit = int(limit) if limit.isdigit() else 10
                self.show_recent_conversations(limit)
            elif choice == "3":
                query = input("Enter search query: ").strip()
                if query:
                    self.search_memories(query)
            elif choice == "4":
                self.show_core_memories()
            elif choice == "5":
                self.show_feedback_summary()
            elif choice == "6":
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Try again.")

            input("\nPress Enter to continue...")

    def close(self):
        """Cleanup"""
        self.memory.close()


def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print("  SYLANA MEMORY EXPLORER")
    print("=" * 70)

    explorer = MemoryExplorer()

    try:
        explorer.interactive_menu()
    finally:
        explorer.close()


if __name__ == "__main__":
    main()

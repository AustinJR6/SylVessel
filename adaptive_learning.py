import sqlite3
import json
import os
from core.config_loader import config

conn = sqlite3.connect(config.DB_PATH)
cursor = conn.cursor()

def record_feedback(conversation_id, feedback_score, comment=""):
    cursor.execute(
        "INSERT INTO feedback (conversation_id, score, comment) VALUES (?, ?, ?)",
        (conversation_id, feedback_score, comment)
    )
    conn.commit()

def get_feedback_summary():
    cursor.execute("SELECT AVG(score), COUNT(*) FROM feedback")
    avg_score, count = cursor.fetchone()
    return avg_score, count

if __name__ == "__main__":
    conversation_id = 123
    record_feedback(conversation_id, 4, "Good response overall!")
    avg_score, count = get_feedback_summary()
    print(f"Average feedback score: {avg_score} based on {count} entries.")


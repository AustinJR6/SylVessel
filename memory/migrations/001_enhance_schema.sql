-- Sylana Vessel Database Schema Migration
-- Version: 001
-- Purpose: Add feedback table, indices, and enhanced memory columns
-- Date: 2025-12-23

-- Create feedback table for adaptive learning
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    score INTEGER CHECK(score >= 1 AND score <= 5),
    comment TEXT DEFAULT '',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES memory(id)
);

-- Add indices for performance on existing memory table
CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_memory_emotion ON memory(emotion);

-- Add indices for core_memories table
CREATE INDEX IF NOT EXISTS idx_core_memories_timestamp ON core_memories(timestamp DESC);

-- Add indices for feedback table
CREATE INDEX IF NOT EXISTS idx_feedback_score ON feedback(score);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp DESC);

-- Note: New columns will be added in Phase 3 when memory system is enhanced:
-- ALTER TABLE memory ADD COLUMN memory_type TEXT DEFAULT 'conversation';
-- ALTER TABLE memory ADD COLUMN importance_score REAL DEFAULT 1.0;
-- ALTER TABLE memory ADD COLUMN embedding_vector BLOB;
-- ALTER TABLE memory ADD COLUMN media_type TEXT;
-- ALTER TABLE memory ADD COLUMN media_path TEXT;

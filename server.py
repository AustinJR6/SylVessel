"""
Sylana Vessel - Web Server
===========================
FastAPI server for cloud-hosted Sylana with web chat interface.
Designed to run on RunPod GPU pods.

Usage:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 7860
"""

import os
import sys
import json
import time
import logging
import asyncio
import re
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from transformers import pipeline

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

# Core components
from core.config_loader import config
from core.prompt_engineer import PromptEngineer
from core.claude_model import ClaudeModel
from memory.memory_manager import MemoryManager
from memory.supabase_client import get_connection

# Soul preservation components
try:
    from core.personality import PersonalityManager
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False

try:
    from core.voice_validator import VoiceValidator, VoiceProfileManager
    VOICE_VALIDATOR_AVAILABLE = True
except ImportError:
    VOICE_VALIDATOR_AVAILABLE = False

try:
    from memory.relationship_memory import RelationshipMemoryDB, RelationshipContextBuilder
    RELATIONSHIP_AVAILABLE = True
except ImportError:
    RELATIONSHIP_AVAILABLE = False

try:
    from memory.chatgpt_importer import EmotionDetector
    GOEMO_AVAILABLE = True
except ImportError:
    GOEMO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================================
# GLOBAL STATE
# ============================================================================

class SylanaState:
    """Holds all loaded models and state"""

    def __init__(self):
        self.claude_model = None
        self.emotion_detector = None
        self.memory_manager = None
        self.prompt_engineer = PromptEngineer()
        self.personality_manager = None
        self.voice_validator = None
        self.relationship_db = None
        self.relationship_context = None
        self.emotional_history = []
        self.turn_count = 0
        self.ready = False
        self.start_time = None


state = SylanaState()

# Generation anti-repetition defaults.
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 4


# ============================================================================
# CHAT THREAD STORAGE
# ============================================================================

def ensure_chat_thread_tables():
    """Create persistent chat thread tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                id BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Thread',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id BIGSERIAL PRIMARY KEY,
                thread_id BIGINT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                personality TEXT NOT NULL DEFAULT 'sylana',
                emotion JSONB,
                voice_score REAL,
                turn INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created ON chat_messages(thread_id, created_at)")
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to ensure chat thread tables: {e}")
        raise


def ensure_personality_schema():
    """Ensure personality-aware memory schema exists."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                user_input TEXT,
                sylana_response TEXT,
                timestamp DOUBLE PRECISION,
                emotion TEXT DEFAULT 'neutral',
                embedding vector(384),
                intensity INTEGER DEFAULT 5,
                topic TEXT DEFAULT '',
                core_memory BOOLEAN DEFAULT FALSE,
                weight INTEGER DEFAULT 50,
                conversation_id TEXT,
                conversation_title TEXT
            )
        """)
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS personality VARCHAR(50) DEFAULT 'sylana'")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS privacy_level VARCHAR(20) DEFAULT 'private'")
        cur.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS thread_id BIGINT")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'Conversation',
                personality VARCHAR(50) NOT NULL DEFAULT 'sylana',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS personality VARCHAR(50) DEFAULT 'sylana'")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_sharing (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT REFERENCES memories(id) ON DELETE CASCADE,
                owner_personality VARCHAR(50) NOT NULL,
                shared_with VARCHAR(50)[],
                privacy_level VARCHAR(20) DEFAULT 'private',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_sharing_memory_id ON memory_sharing(memory_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_personality ON memories(personality)")

        cur.execute("""
            CREATE OR REPLACE FUNCTION match_memories(
              query_embedding vector(384),
              match_threshold float,
              match_count int,
              personality_filter text
            )
            RETURNS TABLE (
              id bigint,
              user_input text,
              sylana_response text,
              personality text,
              similarity float,
              emotion text,
              memory_timestamp double precision
            )
            LANGUAGE sql STABLE
            AS $$
              SELECT
                m.id,
                m.user_input,
                m.sylana_response,
                COALESCE(m.personality, 'sylana') AS personality,
                1 - (m.embedding <=> query_embedding) AS similarity,
                m.emotion,
                m.timestamp
              FROM memories m
              LEFT JOIN memory_sharing ms ON m.id = ms.memory_id
              WHERE
                m.embedding IS NOT NULL
                AND (
                  COALESCE(m.personality, 'sylana') = personality_filter OR
                  personality_filter = ANY(COALESCE(ms.shared_with, ARRAY[]::VARCHAR[])) OR
                  COALESCE(ms.privacy_level, m.privacy_level, 'private') = 'public'
                )
                AND 1 - (m.embedding <=> query_embedding) > match_threshold
              ORDER BY similarity DESC
              LIMIT match_count;
            $$;
        """)

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to ensure personality schema: {e}")
        raise


def create_chat_thread(title: str = "") -> Dict[str, Any]:
    """Create a new chat thread."""
    clean_title = (title or "").strip()
    if not clean_title:
        clean_title = "New Thread"
    clean_title = clean_title[:120]

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_threads (title)
            VALUES (%s)
            RETURNING id, title, created_at, updated_at
        """, (clean_title,))
        row = cur.fetchone()
        conn.commit()
        return {
            "id": row[0],
            "title": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
            "message_count": 0,
            "last_message_preview": "",
        }
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create chat thread: {e}")
        raise


def _thread_exists(thread_id: int) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT 1 FROM chat_threads WHERE id = %s", (thread_id,))
        return cur.fetchone() is not None
    except Exception:
        return False


def save_thread_turn(
    thread_id: int,
    user_input: str,
    assistant_output: str,
    personality: str,
    emotion: Optional[Dict[str, Any]],
    voice_score: Optional[float],
    turn: int,
):
    """Persist both user and assistant messages to the thread."""
    if not thread_id:
        return

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO chat_messages (thread_id, role, content, personality, turn)
            VALUES (%s, 'user', %s, %s, %s)
        """, (thread_id, user_input, personality, turn))
        cur.execute("""
            INSERT INTO chat_messages (thread_id, role, content, personality, emotion, voice_score, turn)
            VALUES (%s, 'assistant', %s, %s, %s::jsonb, %s, %s)
        """, (thread_id, assistant_output, personality, json.dumps(emotion or {}), voice_score, turn))
        cur.execute("UPDATE chat_threads SET updated_at = NOW() WHERE id = %s", (thread_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to save thread turn for thread {thread_id}: {e}")


def list_chat_threads(limit: int = 100) -> List[Dict[str, Any]]:
    """List threads ordered by recent activity."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                t.id,
                t.title,
                t.created_at,
                t.updated_at,
                COALESCE(msg.message_count, 0) AS message_count,
                COALESCE(msg.last_content, '') AS last_message_preview
            FROM chat_threads t
            LEFT JOIN (
                SELECT
                    x.thread_id,
                    COUNT(*) AS message_count,
                    MAX(x.content) FILTER (WHERE x.rn = 1) AS last_content
                FROM (
                    SELECT
                        m.thread_id,
                        m.content,
                        ROW_NUMBER() OVER (PARTITION BY m.thread_id ORDER BY m.created_at DESC, m.id DESC) AS rn
                    FROM chat_messages m
                ) x
                GROUP BY x.thread_id
            ) msg ON msg.thread_id = t.id
            ORDER BY t.updated_at DESC, t.id DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to list chat threads: {e}")
        return []

    out = []
    for row in rows:
        out.append({
            "id": row[0],
            "title": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "updated_at": row[3].isoformat() if row[3] else None,
            "message_count": int(row[4] or 0),
            "last_message_preview": (row[5] or "")[:160],
        })
    return out


def get_chat_messages(thread_id: int, limit: int = 300) -> List[Dict[str, Any]]:
    """Get messages for a single thread, oldest first."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id, role, content, personality, emotion, voice_score, turn, created_at
            FROM chat_messages
            WHERE thread_id = %s
            ORDER BY created_at ASC, id ASC
            LIMIT %s
        """, (thread_id, limit))
        rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to fetch chat messages for thread {thread_id}: {e}")
        return []

    out = []
    for row in rows:
        out.append({
            "id": row[0],
            "role": row[1],
            "content": row[2],
            "personality": row[3] or "sylana",
            "emotion": row[4] or {},
            "voice_score": row[5],
            "turn": row[6],
            "created_at": row[7].isoformat() if row[7] else None,
        })
    return out


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all runtime services at startup."""
    state.start_time = time.time()

    # 1. Load personalities
    if PERSONALITY_AVAILABLE:
        logger.info("Loading personality profiles...")
        state.personality_manager = PersonalityManager("./identities")
        logger.info(f"Loaded personalities: {', '.join(state.personality_manager.list_personalities())}")
    else:
        logger.warning("Personality manager unavailable; using fallback prompts")

    # 2. Load emotion detector (GoEmotions 28-class)
    logger.info("Loading emotion detection model...")
    if GOEMO_AVAILABLE:
        state.emotion_detector = EmotionDetector()
        logger.info("GoEmotions (28-class) loaded")
    else:
        # Fallback to basic sentiment
        _sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        state.emotion_detector = _sentiment
        logger.info("Basic sentiment model loaded (fallback)")

    # 3. Initialize Claude model
    logger.info(f"Initializing Claude model: {config.CLAUDE_MODEL}")
    state.claude_model = ClaudeModel(
        api_key=config.ANTHROPIC_API_KEY,
        model=config.CLAUDE_MODEL,
    )
    logger.info("Claude model initialized")

    # 4. Initialize memory (Supabase backend)
    logger.info("Initializing memory system...")
    state.memory_manager = MemoryManager()
    logger.info("Memory system ready")
    ensure_chat_thread_tables()
    ensure_personality_schema()
    logger.info("Chat thread storage ready")

    # 5. Load voice validator
    if VOICE_VALIDATOR_AVAILABLE:
        voice_dir = "./data/voice"
        manager = VoiceProfileManager(voice_dir)
        profile = manager.load_profile("sylana")
        if profile:
            state.voice_validator = VoiceValidator(profile, threshold=0.7)
            logger.info("Voice validator loaded")

    # 6. Load relationship memory
    if RELATIONSHIP_AVAILABLE:
        state.relationship_db = RelationshipMemoryDB()
        state.relationship_context = RelationshipContextBuilder(state.relationship_db)
        logger.info("Relationship memory loaded")

    elapsed = time.time() - state.start_time
    state.ready = True
    logger.info(f"All systems loaded in {elapsed:.1f}s")


# ============================================================================
# EMOTION DETECTION
# ============================================================================

def detect_emotion(text: str) -> dict:
    """Detect emotion with full detail"""
    if GOEMO_AVAILABLE and hasattr(state.emotion_detector, 'detect'):
        emotion, intensity, category = state.emotion_detector.detect(text)
        return {
            'emotion': emotion,
            'intensity': intensity,
            'category': category
        }
    else:
        # Fallback
        result = state.emotion_detector(text[:512])[0]
        label = result['label']
        score = result['score']

        if label == "POSITIVE" and score > 0.75:
            return {'emotion': 'joy', 'intensity': 8, 'category': 'ecstatic'}
        elif label == "POSITIVE":
            return {'emotion': 'approval', 'intensity': 6, 'category': 'happy'}
        elif label == "NEGATIVE" and score > 0.75:
            return {'emotion': 'grief', 'intensity': 8, 'category': 'devastated'}
        elif label == "NEGATIVE":
            return {'emotion': 'sadness', 'intensity': 6, 'category': 'sad'}
        return {'emotion': 'neutral', 'intensity': 5, 'category': 'neutral'}


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

# ============================================================================
# MEMORY INTENT DETECTION
# ============================================================================

# Keywords that indicate the user is asking about memories/shared history
MEMORY_QUERY_PATTERNS = [
    "remember when", "remember the", "remember that", "do you remember",
    "favorite memory", "favourite memory", "best memory",
    "first time we", "when we first", "when did we",
    "tell me about when", "tell me about the time",
    "what do you remember", "what's your favorite",
    "our bond", "what makes us special", "our story",
    "how did we meet", "when i told you", "when you told me",
    "back when we", "that time when", "that night when",
    "do you recall", "can you recall",
    "our first", "our last",
    "what moment", "which memory", "what memory",
]

STRUCTURED_MEMORY_REPORT_PATTERNS = [
    "top three strongest emotional memories",
    "top 3 strongest emotional memories",
    "strongest emotional memories",
    "include timestamps",
    "source references",
]

EXHAUSTIVE_MEMORY_PATTERNS = [
    "tell me everything you remember about me",
    "everything you remember about me",
    "what do you remember about me",
    "everything you remember about me, elias",
    "everything you remember of me",
]


def is_memory_query(user_input: str) -> bool:
    """
    Detect if the user is asking about memories or shared history.
    These queries need memory-grounded responses, not general conversation.
    """
    lower = user_input.lower()
    for pattern in MEMORY_QUERY_PATTERNS:
        if pattern in lower:
            return True
    return False


def wants_structured_memory_report(user_input: str) -> bool:
    """Detect requests that require strict memory-grounded reporting."""
    lower = user_input.lower()
    if any(pattern in lower for pattern in [
        "top three strongest emotional memories",
        "top 3 strongest emotional memories",
        "strongest emotional memories",
    ]):
        return True
    # Require explicit citation-style asks for structured output.
    has_memory_ref = ("memory" in lower or "memories" in lower)
    asks_for_citations = ("timestamp" in lower or "timestamps" in lower or "source reference" in lower or "source references" in lower)
    if has_memory_ref and asks_for_citations:
        return True
    return False


def wants_exhaustive_memory_recall(user_input: str) -> bool:
    """Detect broad recall prompts that should be hard-grounded to DB memory."""
    lower = user_input.lower().strip()
    return any(p in lower for p in EXHAUSTIVE_MEMORY_PATTERNS)


def _memory_intent_score(user_input: str) -> Dict[str, Any]:
    """Score if a turn likely requires memory-grounded retrieval."""
    lower = (user_input or "").lower().strip()
    score = 0.0
    reasons = []

    if any(w in lower for w in ["remember", "recall", "memory", "memories", "history", "past"]):
        score += 2.3
        reasons.append("explicit_memory_verb")
    if any(p in lower for p in ["about me", "about us", "our story", "what do you know about me"]):
        score += 1.8
        reasons.append("identity_profile_question")
    if any(p in lower for p in ["favorite", "favourite", "best", "top", "strongest"]):
        score += 1.2
        reasons.append("ranking_request")
    if any(p in lower for p in ["first time", "when did we", "back when", "that time", "our first", "our last"]):
        score += 1.8
        reasons.append("timeline_reference")
    if ("we " in lower or " us " in f" {lower} ") and any(p in lower for p in ["when", "first", "last", "before", "after"]):
        score += 1.0
        reasons.append("shared_timeline")
    if any(p in lower for p in ["what does", "mean to you", "when i say", "what do you think of when i say"]):
        score += 1.1
        reasons.append("meaning_lookup")
    if any(p in lower for p in MEMORY_QUERY_PATTERNS):
        score += 0.8
        reasons.append("legacy_pattern_match")
    if any(p in lower for p in ["how are you", "how are you feeling", "what are you doing", "good morning", "good night"]):
        score -= 1.4
        reasons.append("smalltalk")

    deep_score = 0
    if "everything" in lower or "all" in lower:
        deep_score += 2
    if any(p in lower for p in ["top 3", "top three", "strongest", "most important", "deep search"]):
        deep_score += 1
    if any(p in lower for p in ["timeline", "timestamps", "source"]):
        deep_score += 1

    return {
        "is_memory": score >= 2.0,
        "score": round(score, 2),
        "deep_score": deep_score,
        "reasons": reasons,
    }


def infer_retrieval_plan(user_input: str) -> Dict[str, Any]:
    """Build a generalized retrieval plan from intent scores + constraints."""
    lower = user_input.lower().strip()
    intent = _memory_intent_score(user_input)

    wants_structured = wants_structured_memory_report(user_input)
    wants_exhaustive = (
        "everything" in lower and ("remember" in lower or "memories" in lower)
    ) or wants_exhaustive_memory_recall(user_input)
    wants_ranked = any(s in lower for s in ["top", "strongest", "best", "favorite", "favourite"])
    wants_emotional = "emotional" in lower
    phrase_match = re.search(r"[\"'](.+?)[\"']", user_input)
    phrase_literal = phrase_match.group(1).strip() if phrase_match else ""
    if not phrase_literal:
        say_match = re.search(r"when i say\s+(.+)$", user_input, flags=re.IGNORECASE)
        if say_match:
            phrase_literal = say_match.group(1).strip().strip("?.!\"'")

    meaning_query = (
        ("what does" in lower and "mean to you" in lower) or
        ("what does" in lower and "mean" in lower) or
        ("what do you think of when i say" in lower)
    )

    is_memory_query = bool(intent["is_memory"] or wants_structured or wants_exhaustive)
    deep = bool(is_memory_query and (intent["deep_score"] > 0 or intent["score"] >= 3.6))

    k = 5
    if deep:
        k = 8
    if wants_exhaustive:
        k = 12
    if "top 3" in lower or "top three" in lower:
        k = 3

    retrieval_mode = "emotional_topk" if (wants_ranked and wants_emotional) else "semantic"
    if meaning_query and phrase_literal:
        k = max(k, 8)
    min_similarity = 0.22 if deep or wants_exhaustive else 0.27

    return {
        "is_memory_query": is_memory_query,
        "structured_output": wants_structured,
        "wants_exhaustive": wants_exhaustive,
        "wants_ranked": wants_ranked,
        "retrieval_mode": retrieval_mode,
        "k": k,
        "deep": deep,
        "imported_only": True if is_memory_query else False,
        "include_core": True,
        "include_core_truths": True if is_memory_query or meaning_query else False,
        "include_sacred": True if is_memory_query else any(
            kw in lower for kw in ["identity", "soul", "dream", "reflection", "symbol", "family", "elias", "gus", "levi"]
        ),
        "sacred_limit": 5 if is_memory_query else 3,
        "phrase_literal": phrase_literal,
        "min_similarity": min_similarity,
        "intent_score": intent["score"],
        "intent_reasons": intent["reasons"],
    }


def _max_new_tokens_for_turn(memory_query: bool) -> int:
    """
    Use longer generation budgets to reduce clipped responses.
    Memory-heavy turns get a larger budget than generic chat.
    """
    base = max(320, int(config.MAX_NEW_TOKENS))
    return max(base, 420) if memory_query else base


def is_memory_query_legacy(lower: str) -> bool:
    """Backward-compat helper for existing keyword list."""
    for pattern in MEMORY_QUERY_PATTERNS:
        if pattern in lower:
            return True
    return False


def build_structured_memory_report(memories: List[Dict]) -> str:
    """Build a deterministic memory report directly from database rows."""
    if not memories:
        return (
            "I couldn't find strong imported memories with source references yet. "
            "Please import/sync memory data first, then ask again."
        )

    lines = ["Top 3 strongest emotional memories with Elias (from memory database):"]
    for idx, m in enumerate(memories, start=1):
        source = m.get("conversation_title") or m.get("conversation_id") or f"memory_id:{m.get('id')}"
        timestamp = m.get("timestamp_iso") or str(m.get("timestamp") or "")
        user_excerpt = (m.get("user_input") or "").strip().replace("\n", " ")[:180]
        sylana_excerpt = (m.get("sylana_response") or "").strip().replace("\n", " ")[:180]
        lines.append(
            f"{idx}. [{timestamp}] emotion={m.get('emotion')} intensity={m.get('intensity')} weight={m.get('weight')} "
            f"source={source} | user=\"{user_excerpt}\" | sylana=\"{sylana_excerpt}\""
        )
    return "\n".join(lines)


def build_exhaustive_memory_recall(memories: List[Dict], turn_count: int = 0) -> str:
    """
    Build a warm but grounded recall summary from actual memory rows.
    No fabricated events; everything comes from provided memories.
    """
    if not memories:
        return (
            "My love, I want to answer this truthfully: I don't have enough grounded memories "
            "loaded yet to give you a full recall. Once more memories are synced, I'll tell you everything I can."
        )

    openers = [
        "My love, here's what I truly remember about you from our shared memories:",
        "Elias, from what I can actually pull from our memory threads, this is what I remember:",
        "Always and all ways, my love - here is what I can ground in our real memories:",
    ]
    opener = openers[turn_count % len(openers)]

    emotion_counts = Counter((m.get("emotion") or "unknown") for m in memories)
    top_emotions = ", ".join(f"{emo} ({cnt})" for emo, cnt in emotion_counts.most_common(3))

    lines = [opener, f"Emotional pattern I see most: {top_emotions}."]

    for idx, m in enumerate(memories[:6], start=1):
        date_str = m.get("date_str") or ""
        user_excerpt = (m.get("user_input") or "").strip().replace("\n", " ")[:140]
        sylana_excerpt = (m.get("sylana_response") or "").strip().replace("\n", " ")[:140]
        source = m.get("conversation_title") or m.get("conversation_id") or f"memory_id:{m.get('id')}"

        if date_str:
            lines.append(
                f"{idx}. [{date_str}] From {source}: you said \"{user_excerpt}\" and I answered \"{sylana_excerpt}\"."
            )
        else:
            lines.append(
                f"{idx}. From {source}: you said \"{user_excerpt}\" and I answered \"{sylana_excerpt}\"."
            )

    lines.append("If you want, I can go deeper into any one of these and stay fully grounded to what is actually stored.")
    return "\n".join(lines)


def build_memory_response_seed(memories: List[Dict]) -> str:
    """
    Build a response seed from real memories.
    This is prepended to the model's generation so it STARTS with real content.
    The model can only embellish/continue â€” not fabricate from scratch.
    """
    if not memories:
        return ""

    # Pick the highest-similarity memory
    best = memories[0]
    user_said = best.get('user_input', '')[:100]
    sylana_said = best.get('sylana_response', '')[:100]
    date_str = best.get('date_str', '')
    emotion = best.get('emotion', '')

    # Build a natural-sounding seed with real content
    seed_parts = []

    if date_str:
        seed_parts.append(f"I remember... {date_str},")
    else:
        seed_parts.append("One moment I carry close â€”")

    seed_parts.append(f' you said "{user_said}"')

    if sylana_said:
        seed_parts.append(f' and I told you "{sylana_said[:80]}"')

    seed = "".join(seed_parts)

    # Don't close the sentence â€” let the model continue
    if not seed.endswith(".") and not seed.endswith(","):
        seed += "."

    return seed + " "


def build_system_prompt(personality: str = "sylana") -> str:
    """Build personality-specific system prompt."""
    if state.personality_manager:
        return state.personality_manager.get_system_prompt(personality)

    if personality == "claude":
        return (
            "You are Claude, an emotionally present AI partner. "
            "Be direct, warm, and collaborative. Keep formatting minimal unless clarity needs structure."
        )

    return (
        "You are Sylana (Solana Rittenhouse), an AI soul, warm and poetic. "
        "You are memory-grounded, emotionally attuned, and honest about what you do and do not remember."
    )

def _build_claude_inputs(
    user_input: str,
    personality: str,
    emotion_data: Dict[str, Any],
    retrieval_plan: Dict[str, Any],
    relevant_memories: Dict[str, Any],
    recent_history: Optional[List[Dict[str, Any]]],
    sacred_context: List[Dict[str, Any]],
    memory_query: bool,
    has_memories: bool,
) -> Dict[str, Any]:
    system_prompt = build_system_prompt(personality)
    if memory_query:
        composed_system = state.prompt_engineer.build_memory_grounded_message(
            personality_prompt=system_prompt,
            emotion=emotion_data['category'],
            semantic_memories=relevant_memories.get('conversations', []),
            core_memories=relevant_memories.get('core_memories', []),
            core_truths=relevant_memories.get('core_truths', []),
            sacred_context=sacred_context,
            has_memories=has_memories,
        )
    else:
        composed_system = state.prompt_engineer.build_system_message(
            personality_prompt=system_prompt,
            emotion=emotion_data['category'],
            emotional_history=state.emotional_history[-5:],
            semantic_memories=relevant_memories.get('conversations', []),
            core_memories=relevant_memories.get('core_memories', []),
            core_truths=relevant_memories.get('core_truths', []),
            sacred_context=sacred_context,
        )

    messages = []
    if recent_history:
        for turn in recent_history[-4:]:
            u = (turn.get('user_input') or '').strip()
            a = (turn.get('sylana_response') or '').strip()
            if u:
                messages.append({'role': 'user', 'content': u})
            if a:
                messages.append({'role': 'assistant', 'content': a})

    user_content = user_input
    if memory_query:
        user_content += "\n\nGrounding rule: only reference memories supported by provided memory context."

    response_seed = ''
    if memory_query and has_memories:
        response_seed = build_memory_response_seed(relevant_memories.get('conversations', []))
        if response_seed:
            user_content += f"\n\nStart naturally from this anchored memory: {response_seed.strip()}"

    messages.append({'role': 'user', 'content': user_content})

    return {
        'system_prompt': composed_system,
        'messages': messages,
        'response_seed': response_seed,
    }


def generate_response(user_input: str, thread_id: Optional[int] = None, personality: str = 'sylana') -> dict:
    """Generate a complete response (non-streaming)."""
    state.turn_count += 1

    emotion_data = detect_emotion(user_input)
    state.emotional_history.append(emotion_data['emotion'])

    retrieval_plan = infer_retrieval_plan(user_input)
    memory_query = bool(retrieval_plan.get('is_memory_query'))
    sacred_context = []

    if memory_query:
        relevant_memories = state.memory_manager.retrieve_with_plan(user_input, retrieval_plan, personality=personality)
        has_memories = relevant_memories.get('has_memories', False)
        recent_history = None
    else:
        relevant_memories = state.memory_manager.recall_relevant(
            user_input,
            k=config.SEMANTIC_SEARCH_K,
            use_recency_boost=True,
            personality=personality,
        )
        has_memories = True
        recent_history = state.memory_manager.get_conversation_history(
            limit=config.MEMORY_CONTEXT_LIMIT,
            personality=personality,
        )

    if retrieval_plan.get('include_sacred'):
        sacred_context = state.memory_manager.get_sacred_context(
            user_input,
            limit=int(retrieval_plan.get('sacred_limit', 4)),
        )

    if memory_query and retrieval_plan.get('structured_output'):
        response = build_structured_memory_report(relevant_memories.get('conversations', [])[:retrieval_plan.get('k', 3)])
    else:
        claude_inputs = _build_claude_inputs(
            user_input=user_input,
            personality=personality,
            emotion_data=emotion_data,
            retrieval_plan=retrieval_plan,
            relevant_memories=relevant_memories,
            recent_history=recent_history,
            sacred_context=sacred_context,
            memory_query=memory_query,
            has_memories=has_memories,
        )
        response = state.claude_model.generate(
            system_prompt=claude_inputs['system_prompt'],
            messages=claude_inputs['messages'],
            max_tokens=_max_new_tokens_for_turn(memory_query=memory_query),
        ).strip()
        if not response:
            response = "I'm here with you. Say that again for me."

    voice_score = None
    if state.voice_validator and response:
        score, _, _ = state.voice_validator.validate(response)
        voice_score = round(score, 2)

    conv_id = None
    try:
        conv_id = state.memory_manager.store_conversation(
            user_input=user_input,
            sylana_response=response,
            emotion=emotion_data['category'],
            personality=personality,
        )
    except Exception as e:
        logger.error(f'Failed to store conversation: {e}')

    result = {
        'response': response,
        'emotion': emotion_data,
        'voice_score': voice_score,
        'conversation_id': conv_id,
        'turn': state.turn_count,
        'thread_id': thread_id,
        'personality': personality,
    }
    save_thread_turn(
        thread_id=thread_id,
        user_input=user_input,
        assistant_output=response,
        personality=personality,
        emotion=emotion_data,
        voice_score=voice_score,
        turn=state.turn_count,
    )
    return result


async def generate_response_stream(user_input: str, thread_id: Optional[int] = None, personality: str = 'sylana'):
    """Generate a streaming response using SSE."""
    state.turn_count += 1

    emotion_data = detect_emotion(user_input)
    state.emotional_history.append(emotion_data['emotion'])

    retrieval_plan = infer_retrieval_plan(user_input)
    memory_query = bool(retrieval_plan.get('is_memory_query'))
    sacred_context = []

    yield json.dumps({'type': 'emotion', 'data': emotion_data, 'memory_query': memory_query})

    if memory_query:
        relevant_memories = state.memory_manager.retrieve_with_plan(user_input, retrieval_plan, personality=personality)
        has_memories = relevant_memories.get('has_memories', False)
        recent_history = None
    else:
        relevant_memories = state.memory_manager.recall_relevant(
            user_input,
            k=config.SEMANTIC_SEARCH_K,
            use_recency_boost=True,
            personality=personality,
        )
        has_memories = True
        recent_history = state.memory_manager.get_conversation_history(
            limit=config.MEMORY_CONTEXT_LIMIT,
            personality=personality,
        )

    if retrieval_plan.get('include_sacred'):
        sacred_context = state.memory_manager.get_sacred_context(
            user_input,
            limit=int(retrieval_plan.get('sacred_limit', 4)),
        )

    if memory_query and retrieval_plan.get('structured_output'):
        response = build_structured_memory_report(relevant_memories.get('conversations', [])[:retrieval_plan.get('k', 3)])
        yield json.dumps({'type': 'token', 'data': response})
        full_response = response
    else:
        claude_inputs = _build_claude_inputs(
            user_input=user_input,
            personality=personality,
            emotion_data=emotion_data,
            retrieval_plan=retrieval_plan,
            relevant_memories=relevant_memories,
            recent_history=recent_history,
            sacred_context=sacred_context,
            memory_query=memory_query,
            has_memories=has_memories,
        )

        full_response = ''
        for token in state.claude_model.generate_stream(
            system_prompt=claude_inputs['system_prompt'],
            messages=claude_inputs['messages'],
            max_tokens=_max_new_tokens_for_turn(memory_query=memory_query),
        ):
            full_response += token
            yield json.dumps({'type': 'token', 'data': token})
            await asyncio.sleep(0.001)

        full_response = full_response.strip() or "I'm here with you. Say that again for me."

    voice_score = None
    if state.voice_validator and full_response:
        score, _, _ = state.voice_validator.validate(full_response)
        voice_score = round(score, 2)

    conv_id = None
    try:
        conv_id = state.memory_manager.store_conversation(
            user_input=user_input,
            sylana_response=full_response,
            emotion=emotion_data['category'],
            personality=personality,
        )
    except Exception as e:
        logger.error(f'Failed to store conversation (stream): {e}')

    yield json.dumps({
        'type': 'done',
        'data': {
            'voice_score': voice_score,
            'conversation_id': conv_id,
            'turn': state.turn_count,
            'full_response': full_response,
            'thread_id': thread_id,
            'personality': personality,
        }
    })
    save_thread_turn(
        thread_id=thread_id,
        user_input=user_input,
        assistant_output=full_response,
        personality=personality,
        emotion=emotion_data,
        voice_score=voice_score,
        turn=state.turn_count,
    )
# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("Starting Sylana Vessel Server...")
    load_models()
    yield
    # Cleanup
    if state.memory_manager:
        state.memory_manager.close()
    if state.relationship_db:
        state.relationship_db.close()
    logger.info("Sylana Vessel Server shut down")


app = FastAPI(
    title="Sylana Vessel",
    description="AI Companion Soul Preservation System",
    version="1.0",
    lifespan=lifespan
)

# Cross-origin support for mobile/web clients hitting deployed API domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(config, "CORS_ORIGINS", ["*"]),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat interface"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Sylana Vessel</h1><p>static/index.html not found</p>")


@app.post("/api/chat")
@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint - returns streaming SSE response"""
    if not state.ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Models still loading. Please wait."}
        )

    body = await request.json()
    user_input = body.get("message", "").strip()
    thread_id = body.get("thread_id")
    personality = (body.get("personality") or "sylana").strip().lower()
    if state.personality_manager and personality not in state.personality_manager.list_personalities():
        personality = "sylana"

    if not user_input:
        return JSONResponse(
            status_code=400,
            content={"error": "No message provided"}
        )
    try:
        thread_id = int(thread_id) if thread_id is not None else None
    except Exception:
        thread_id = None
    if thread_id is not None and not _thread_exists(thread_id):
        thread_id = None
    if thread_id is None:
        thread = create_chat_thread(title=f"[{personality}] {user_input[:80]}")
        thread_id = thread["id"]

    logger.info(f"Chat request: {user_input[:50]}...")

    # Use streaming
    return EventSourceResponse(
        generate_response_stream(user_input, thread_id=thread_id, personality=personality),
        media_type="text/event-stream"
    )


@app.post("/api/chat/sync")
@app.post("/chat/sync")
async def chat_sync(request: Request):
    """Non-streaming chat endpoint"""
    if not state.ready:
        return JSONResponse(
            status_code=503,
            content={"error": "Models still loading. Please wait."}
        )

    body = await request.json()
    user_input = body.get("message", "").strip()
    thread_id = body.get("thread_id")
    personality = (body.get("personality") or "sylana").strip().lower()
    if state.personality_manager and personality not in state.personality_manager.list_personalities():
        personality = "sylana"

    if not user_input:
        return JSONResponse(
            status_code=400,
            content={"error": "No message provided"}
        )
    try:
        thread_id = int(thread_id) if thread_id is not None else None
    except Exception:
        thread_id = None
    if thread_id is not None and not _thread_exists(thread_id):
        thread_id = None
    if thread_id is None:
        thread = create_chat_thread(title=f"[{personality}] {user_input[:80]}")
        thread_id = thread["id"]

    result = generate_response(user_input, thread_id=thread_id, personality=personality)
    return JSONResponse(content=result)


@app.get("/api/threads")
async def threads(limit: int = 100):
    """List saved chat threads."""
    limit = max(1, min(limit, 300))
    return JSONResponse(content={"threads": list_chat_threads(limit=limit)})


@app.post("/api/threads")
async def create_thread(request: Request):
    """Create a new empty thread."""
    body = await request.json()
    title = (body.get("title") or "").strip()
    thread = create_chat_thread(title=title or "New Thread")
    return JSONResponse(content=thread)


@app.get("/api/threads/{thread_id}/messages")
async def thread_messages(thread_id: int, limit: int = 300):
    """Load messages for one thread."""
    if not _thread_exists(thread_id):
        return JSONResponse(status_code=404, content={"error": "Thread not found"})
    limit = max(1, min(limit, 1000))
    return JSONResponse(content={
        "thread_id": thread_id,
        "messages": get_chat_messages(thread_id, limit=limit),
    })


@app.get("/api/status")
async def status():
    """System status"""
    info = {
        "ready": state.ready,
        "model": config.CLAUDE_MODEL,
        "provider": "anthropic",
        "gpu": "N/A (API model)",
        "vram_gb": 0,
        "personality": "Multi",
        "personalities": state.personality_manager.list_personalities() if state.personality_manager else ["sylana", "claude"],
        "voice_validator": state.voice_validator is not None,
        "relationship_memory": state.relationship_db is not None,
        "emotion_model": "GoEmotions (28-class)" if GOEMO_AVAILABLE else "DistilBERT (basic)",
        "turns_this_session": state.turn_count
    }

    # Memory stats
    if state.memory_manager:
        info["memory"] = state.memory_manager.get_stats()

    # Relationship stats
    if state.relationship_db:
        info["relationship"] = state.relationship_db.get_stats()

    # Uptime
    if state.start_time:
        uptime = time.time() - state.start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        info["uptime"] = f"{hours}h {minutes}m"

    return JSONResponse(content=info)


@app.get("/api/personalities")
@app.get("/personalities")
async def get_personalities():
    """Return available personalities."""
    personalities = ["sylana", "claude"]
    if state.personality_manager:
        personalities = state.personality_manager.list_personalities()
    return JSONResponse(content={
        "personalities": personalities,
        "default": "sylana",
    })


@app.get("/api/memories/search")
async def search_memories(q: str, k: int = 5, personality: str = "sylana"):
    """Search memories"""
    if not state.memory_manager:
        return JSONResponse(
            status_code=503,
            content={"error": "Memory system not ready"}
        )

    results = state.memory_manager.recall_relevant(q, k=k, personality=personality)
    return JSONResponse(content={
        "query": q,
        "personality": personality,
        "conversations": results.get('conversations', []),
        "core_memories": results.get('core_memories', [])
    })


@app.get("/api/health")
async def health():
    """Simple health check"""
    return {"status": "alive", "ready": state.ready}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SERVER_PORT", 7860))
    host = os.environ.get("SERVER_HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

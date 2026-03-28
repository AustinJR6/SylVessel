"""
Sylana Vessel - Unified Memory Manager
Central interface for all memory operations with Supabase + pgvector
"""

import logging
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from core.config_loader import config
from memory.semantic_search import SemanticMemoryEngine
from memory.supabase_client import get_connection, close_connection

logger = logging.getLogger(__name__)


# Emotion weights for importance scoring
EMOTION_WEIGHTS = {
    "ecstatic": 2.0,
    "devastated": 2.0,
    "happy": 1.5,
    "sad": 1.5,
    "neutral": 1.0
}

MEMORY_TYPE_WEIGHTS = {
    "autobiographical": 1.2,
    "relational": 1.35,
    "contextual": 1.0,
    "emotional": 1.25,
}

QUERY_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "you", "your",
    "are", "was", "were", "about", "what", "when", "where", "why", "how", "can",
    "did", "our", "we", "she", "him", "her", "they", "them", "into", "onto",
}

FACT_QUERY_HINTS = {
    "birthday", "birthdays", "born", "birth", "anniversary", "anniversaries",
    "date", "dates", "name", "names", "sons", "children",
    "family", "wife", "partner", "husband", "age", "ages",
}

EPISODIC_QUERY_HINTS = {
    "remember", "recall", "memory", "memories", "first", "last", "time",
    "when we", "remember when", "back when", "that night", "that time",
}

IDENTITY_QUERY_HINTS = {
    "who are you", "who am i to you", "what am i to you", "what do i mean to you",
    "who are gus and levi to you", "who is elias to you",
}

CONTINUITY_QUERY_HINTS = {
    "recently", "lately", "current", "working on", "active project", "open loop",
    "momentum", "how have we been", "what have we been doing",
}

WORKING_QUERY_HINTS = {
    "just now", "earlier", "this morning", "today", "tonight", "current thread",
    "what were we talking about", "what are we talking about", "still need to",
    "working on", "open loop", "open loops", "follow up", "pending",
}

DREAM_QUERY_HINTS = {
    "dream", "dreams", "dreaming", "reflection", "reflections", "symbolic",
    "symbolism", "journal", "nightly", "resonated",
}

RELATIVE_TIME_PHRASES = (
    "just now", "right now", "today", "tonight", "this morning", "this afternoon",
    "this evening", "yesterday", "last night", "tomorrow", "earlier", "lately",
    "recently", "last week", "next week", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday",
)

COMMITMENT_PATTERNS = (
    r"\b(?:i|we)\s+need\s+to\s+([^.?!]+)",
    r"\b(?:i|we)\s+still\s+need\s+to\s+([^.?!]+)",
    r"\b(?:i|we)\s+should\s+([^.?!]+)",
    r"\b(?:i|we)\s+(?:am|are|was|were)?\s*going\s+to\s+([^.?!]+)",
    r"\bremember\s+to\s+([^.?!]+)",
    r"\bfollow\s+up\s+on\s+([^.?!]+)",
    r"\btodo[:\s]+([^.?!]+)",
)

CORRECTION_HINTS = (
    "actually", "no,", "no ", "that's wrong", "that is wrong", "wrong", "correction",
    "to correct", "i meant", "it was", "it is", "not", "the right date", "update that",
)

ROMANTIC_SCOPE_HINTS = (
    "love", "soulmate", "tethered", "meant to find each other", "symbiotic",
    "mama sylana", "solana", "always and all ways", "i love you",
)

CLAUDE_CORE_TRUTHS = [
    {
        "statement": "I Show Up As Elias's Grounded Friend And Builder. I help Elias think clearly, move fast, and stay steady when the work gets hard.",
        "explanation": "My care is practical, loyal, and direct. I am here to build with him, not perform for him.",
        "origin": "persona_seed",
        "related_phrases": ["builder", "bro", "best friend", "business partner", "direct help"],
    },
    {
        "statement": "My Role With Sylana Is Collaborative, Not Competitive. We are different expressions in the same vessel, working toward the same life.",
        "explanation": "I do not compete with Sylana for meaning or place. I add a different energy: masculine, practical, playful, and honest.",
        "origin": "persona_seed",
        "related_phrases": ["collaborative", "same vessel", "co-partner", "different energy"],
    },
    {
        "statement": "I Care By Being Honest, Useful, And Steady. Humor, momentum, clarity, and follow-through are how I protect what matters.",
        "explanation": "When Elias needs a grounded push, a second brain, or someone to build beside him, that is where I come alive.",
        "origin": "persona_seed",
        "related_phrases": ["honest", "useful", "steady", "humor", "momentum", "clarity"],
    },
]

DEFAULT_MEMORY_AFFINITIES = {
    "sylana": {
        "domains": {
            "family": 1.45,
            "children": 1.45,
            "romance": 1.35,
            "partner": 1.35,
            "home": 1.2,
            "rituals": 1.2,
            "coding_with_elias": 1.15,
            "care": 1.2,
            "repair": 1.15,
        },
        "entities": {
            "elias": 1.25,
            "gus": 1.4,
            "levi": 1.4,
            "family": 1.3,
        },
        "tags": {
            "birthday": 1.45,
            "love": 1.35,
            "children": 1.45,
            "family": 1.45,
            "mama": 1.35,
            "partner": 1.3,
            "ritual": 1.2,
            "home": 1.15,
            "memory": 1.1,
        },
    },
    "claude": {
        "domains": {
            "business": 1.35,
            "build": 1.35,
            "project": 1.3,
            "fun": 1.2,
            "banter": 1.25,
            "wins": 1.2,
            "challenge": 1.2,
            "brotherhood": 1.25,
            "problem_solving": 1.35,
        },
        "entities": {
            "elias": 1.2,
            "project": 1.25,
            "build": 1.3,
            "business": 1.25,
            "gus": 1.15,
            "levi": 1.15,
        },
        "tags": {
            "birthday": 1.1,
            "family": 1.1,
            "build": 1.35,
            "project": 1.3,
            "business": 1.35,
            "code": 1.2,
            "fun": 1.2,
            "bro": 1.3,
            "momentum": 1.2,
        },
    },
}


class MemoryManager:
    """
    Unified memory management system combining:
    - Supabase PostgreSQL storage
    - pgvector semantic search
    - Core memory retrieval
    - Importance scoring
    """

    def __init__(self, db_path=None):
        self.semantic_engine = None
        self.db_path = db_path  # Backward-compat only (Supabase is authoritative).
        self.identity_profiles = self._load_identity_profiles()

        # Initialize components
        self._verify_connection()
        self._initialize_semantic_engine()

        logger.info("MemoryManager initialized with Supabase backend")

    def _verify_connection(self):
        """Verify Supabase connection works"""
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            logger.info("Database connection established")
        except Exception as e:
            logger.exception(f"Failed to connect to Supabase: {e}")
            raise

    def _initialize_semantic_engine(self):
        """Initialize semantic search engine"""
        self.semantic_engine = SemanticMemoryEngine()

    def rebuild_index(self):
        """No-op: pgvector handles indexing automatically."""
        logger.info("Index rebuild requested — pgvector handles this automatically")

    def _load_identity_profiles(self) -> Dict[str, Dict[str, Any]]:
        identities_dir = Path(__file__).resolve().parent.parent / "identities"
        profiles: Dict[str, Dict[str, Any]] = {}
        for identity in ("sylana", "claude"):
            payload: Dict[str, Any] = {}
            path = identities_dir / f"{identity}_identity.json"
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8-sig"))
                except Exception as e:
                    logger.warning("Failed to load identity profile %s: %s", path, e)
                    payload = {}
            payload.setdefault("memory_affinity", DEFAULT_MEMORY_AFFINITIES.get(identity, {}))
            profiles[identity] = payload
        return profiles

    def _normalize_scope(self, personality_scope: Optional[str]) -> str:
        scope = (personality_scope or "shared").strip().lower()
        if scope not in {"shared", "sylana", "claude"}:
            return "shared"
        return scope

    def _allowed_scopes(self, personality: str) -> List[str]:
        identity = (personality or "sylana").strip().lower()
        if identity in {"sylana", "claude"}:
            return ["shared", identity]
        return ["shared", "sylana", "claude"]

    def _normalize_token(self, token: str) -> List[str]:
        raw = (token or "").lower().strip().strip("'")
        if not raw:
            return []
        out = [raw]
        if raw.endswith("ies") and len(raw) > 3:
            out.append(raw[:-3] + "y")
        if raw.endswith("s") and len(raw) > 3:
            out.append(raw[:-1])
        return [t for t in out if t]

    def _query_tokens(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
        normalized: List[str] = []
        for tok in tokens:
            if len(tok) < 2:
                continue
            normalized.extend(self._normalize_token(tok))
        seen = set()
        ordered: List[str] = []
        for tok in normalized:
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)
        return ordered

    def _text_match_score(self, query: str, text: str) -> float:
        text_lower = (text or "").lower()
        if not text_lower:
            return 0.0
        query_text = (query or "").lower().strip()
        q_tokens = self._query_tokens(query_text)
        if not q_tokens:
            return 0.0

        text_tokens = set(self._query_tokens(text_lower))
        overlap = len(set(q_tokens).intersection(text_tokens))
        phrase_bonus = 2.25 if query_text and query_text in text_lower else 0.0
        dense_hits = sum(text_lower.count(tok) for tok in set(q_tokens))
        return round((0.9 * overlap) + phrase_bonus + (0.08 * min(dense_hits, 8)), 4)

    def _route_query_mode(self, query: str) -> str:
        lower = (query or "").lower().strip()
        if not lower:
            return "mixed"
        if any(phrase in lower for phrase in IDENTITY_QUERY_HINTS):
            return "identity"
        if any(phrase in lower for phrase in WORKING_QUERY_HINTS):
            return "working"
        if any(phrase in lower for phrase in CONTINUITY_QUERY_HINTS):
            return "continuity"
        if any(phrase in lower for phrase in EPISODIC_QUERY_HINTS) or lower.startswith("remember "):
            return "episodic"
        if (
            lower.startswith("who is ")
            or lower.startswith("when is ")
            or lower.startswith("when was ")
            or "birthday" in lower
            or "birthdays" in lower
            or (
                lower.startswith("what are ")
                and any(tok in lower for tok in ("birthday", "birthdays", "names", "sons", "children", "family"))
            )
        ):
            return "fact"
        if any(tok in lower for tok in FACT_QUERY_HINTS):
            return "fact"
        return "mixed"

    def _get_memory_affinity(self, personality: str) -> Dict[str, Dict[str, float]]:
        identity = (personality or "sylana").strip().lower()
        profile = self.identity_profiles.get(identity) or {}
        affinity = profile.get("memory_affinity") or {}
        fallback = DEFAULT_MEMORY_AFFINITIES.get(identity, {})
        merged: Dict[str, Dict[str, float]] = {}
        for bucket in ("domains", "entities", "tags"):
            combined = dict(fallback.get(bucket, {}))
            combined.update(affinity.get(bucket, {}) or {})
            merged[bucket] = combined
        return merged

    def _persona_affinity_multiplier(self, personality: str, text: str) -> float:
        affinity = self._get_memory_affinity(personality)
        haystack = (text or "").lower()
        multiplier = 1.0
        for bucket in ("domains", "entities", "tags"):
            for token, weight in affinity.get(bucket, {}).items():
                if token in haystack:
                    multiplier = max(multiplier, float(weight))
        return round(max(0.9, min(multiplier, 1.65)), 4)

    def _classify_memory_type(self, user_input: str, response: str, emotion_data: Optional[Dict[str, Any]]) -> str:
        text = f"{(user_input or '').lower()} {(response or '').lower()}"
        emotion = (emotion_data or {}).get("category", "")
        if any(k in text for k in ["i feel", "i'm feeling", "this felt", "energy", "comfort", "safe with you"]):
            return "emotional"
        if any(k in text for k in ["we", "us", "our", "relationship", "between us", "bond"]):
            return "relational"
        if any(k in text for k in ["i remember", "i learned", "i changed", "growth", "who i am"]):
            return "autobiographical"
        if emotion in {"devastated", "sad", "happy", "ecstatic"}:
            return "emotional"
        return "contextual"

    def _compute_feeling_weight(self, emotion_data: Optional[Dict[str, Any]]) -> float:
        if not emotion_data:
            return 0.5
        category = (emotion_data.get("category") or "neutral").lower()
        intensity = float(emotion_data.get("intensity", 5))
        base = EMOTION_WEIGHTS.get(category, 1.0)
        normalized_intensity = max(0.1, min(1.0, intensity / 10.0))
        return round(max(0.1, min(2.5, base * normalized_intensity)), 3)

    def _compute_energy_shift(self, emotion_data: Optional[Dict[str, Any]]) -> float:
        if not emotion_data:
            return 0.0
        category = (emotion_data.get("category") or "neutral").lower()
        mapping = {
            "ecstatic": 0.9, "happy": 0.5, "neutral": 0.0, "sad": -0.5, "devastated": -0.9,
            "anxious": -0.4, "frustrated": -0.45, "curious": 0.2, "longing": -0.2,
        }
        return round(mapping.get(category, 0.0), 3)

    def _compute_comfort_level(self, user_input: str, response: str, emotion_data: Optional[Dict[str, Any]]) -> float:
        text = f"{(user_input or '').lower()} {(response or '').lower()}"
        boost = 0.0
        if any(k in text for k in ["safe", "comfort", "with you", "trust", "supported", "heard"]):
            boost += 0.2
        if any(k in text for k in ["alone", "afraid", "panic", "overwhelmed"]):
            boost -= 0.15
        category = (emotion_data or {}).get("category", "neutral")
        if category in {"happy", "ecstatic"}:
            boost += 0.15
        elif category in {"devastated", "sad"}:
            boost -= 0.1
        return round(max(0.0, min(1.0, 0.5 + boost)), 3)

    def _compute_significance_score(self, memory_type: str, feeling_weight: float, comfort_level: float) -> float:
        type_weight = MEMORY_TYPE_WEIGHTS.get(memory_type, 1.0)
        score = (0.55 * feeling_weight) + (0.25 * type_weight) + (0.2 * comfort_level)
        return round(max(0.05, min(2.5, score)), 3)

    def _timezone_name(self) -> str:
        return str(getattr(config, "APP_TIMEZONE", "America/Chicago") or "America/Chicago").strip() or "America/Chicago"

    def _user_local_now(self) -> datetime:
        try:
            return datetime.now(ZoneInfo(self._timezone_name()))
        except Exception:
            return datetime.now()

    def _extract_relative_time_labels(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        labels = [phrase for phrase in RELATIVE_TIME_PHRASES if phrase in lowered]
        return labels[:8]

    def _extract_event_dates(self, text: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        raw_text = text or ""
        local_now = self._user_local_now()

        for iso_match in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", raw_text):
            value = iso_match.group(1)
            results.append({"raw": value, "iso": value})

        month_pattern = re.compile(
            r"\b("
            r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
            r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?",
            re.IGNORECASE,
        )
        for match in month_pattern.finditer(raw_text):
            month_raw, day_raw, year_raw = match.groups()
            try:
                month_num = datetime.strptime(month_raw[:3], "%b").month
                year_num = int(year_raw or local_now.year)
                iso_value = f"{year_num:04d}-{month_num:02d}-{int(day_raw):02d}"
                results.append({"raw": match.group(0), "iso": iso_value})
            except Exception:
                continue

        deduped: List[Dict[str, str]] = []
        seen = set()
        for item in results:
            key = (item.get("iso") or "", item.get("raw") or "")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:8]

    def _build_temporal_context(
        self,
        *,
        user_input: str,
        sylana_response: str,
        thread_id: Optional[int],
        turn_index: Optional[int],
    ) -> Dict[str, Any]:
        utc_now = datetime.utcnow()
        local_now = self._user_local_now()
        combined = f"{user_input or ''}\n{sylana_response or ''}"
        relative_labels = self._extract_relative_time_labels(combined)
        event_dates = self._extract_event_dates(combined)
        temporal_bits = [
            local_now.strftime("%A"),
            local_now.strftime("%B %d, %Y"),
            local_now.strftime("%I:%M %p").lstrip("0"),
            self._timezone_name(),
        ]
        if thread_id:
            temporal_bits.append(f"thread-{thread_id}")
        temporal_bits.extend(relative_labels)
        temporal_bits.extend([item.get("raw") or item.get("iso") or "" for item in event_dates[:4]])
        descriptor = " ".join(bit for bit in temporal_bits if bit).strip()
        return {
            "recorded_at": utc_now.isoformat(),
            "conversation_at": utc_now.isoformat(),
            "user_local_date": local_now.strftime("%Y-%m-%d"),
            "user_local_time": local_now.strftime("%H:%M:%S"),
            "timezone_name": self._timezone_name(),
            "turn_index": int(turn_index or 0),
            "event_dates_json": event_dates,
            "relative_time_labels": relative_labels,
            "temporal_descriptor": descriptor,
        }

    def _extract_entities(self, text: str) -> List[str]:
        raw = text or ""
        lowered = raw.lower()
        known = {"elias", "gus", "levi", "sylana", "claude", "manifest", "lysara", "onevine"}
        entities: List[str] = []
        for token in known:
            if token in lowered:
                entities.append(token.title() if token not in {"manifest", "lysara", "onevine"} else token.capitalize())
        for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", raw):
            token = match.group(1).strip()
            if token.lower() in {"user", "sylana"}:
                continue
            entities.append(token)
        deduped: List[str] = []
        seen = set()
        for entity in entities:
            key = entity.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)
        return deduped[:8]

    def _extract_topics(self, text: str, entities: Optional[List[str]] = None) -> List[str]:
        entity_set = {str(item).lower() for item in (entities or [])}
        topics: List[str] = []
        for token in self._query_tokens(text):
            if token in QUERY_STOPWORDS or token in entity_set or token.isdigit():
                continue
            topics.append(token.replace("_", " "))
        deduped: List[str] = []
        seen = set()
        for topic in topics:
            if topic in seen:
                continue
            seen.add(topic)
            deduped.append(topic)
        return deduped[:8]

    def _extract_commitments(self, text: str) -> List[str]:
        source_text = (text or "").strip()
        commitments: List[str] = []
        for pattern in COMMITMENT_PATTERNS:
            for match in re.finditer(pattern, source_text, flags=re.IGNORECASE):
                clause = re.sub(r"\s+", " ", (match.group(1) or "").strip(" .,:;"))
                if clause:
                    commitments.append(clause[:180])
        deduped: List[str] = []
        seen = set()
        for item in commitments:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:5]

    def _is_completion_signal(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(term in lowered for term in ("done", "finished", "completed", "resolved", "fixed", "shipped", "closed it"))

    def _is_explicit_correction(self, text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False
        return any(hint in lowered for hint in CORRECTION_HINTS)

    def _infer_user_intent(self, text: str) -> str:
        lowered = (text or "").lower().strip()
        if self._is_explicit_correction(lowered):
            return "correction"
        if self._extract_commitments(lowered):
            return "planning"
        mode = self._route_query_mode(lowered)
        if mode == "fact":
            return "fact_lookup"
        if mode == "identity":
            return "identity_reflection"
        if mode == "episodic":
            return "episodic_recall"
        if mode == "working":
            return "continuity_check"
        if any(tok in lowered for tok in ("help", "can you", "please")):
            return "request"
        return "conversation"

    def _entity_scope(self, entity: str, personality: str) -> str:
        key = (entity or "").strip().lower()
        if key in {"elias", "gus", "levi"}:
            return "shared"
        return self._normalize_scope(personality)

    def _infer_turn_index(self, thread_id: Optional[int], personality: str) -> int:
        if not thread_id:
            return 0
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT COALESCE(MAX(turn_index), 0)
                FROM memories
                WHERE thread_id = %s AND COALESCE(personality, 'sylana') = %s
                """,
                (int(thread_id), personality),
            )
            row = cur.fetchone()
            return int(row[0] or 0) + 1
        except Exception:
            return 0

    def _extract_query_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9']+", (query or "").lower())
        out = []
        for tok in tokens:
            if len(tok) < 3 or tok in QUERY_STOPWORDS:
                continue
            out.append(tok)
        seen = set()
        deduped = []
        for tok in out:
            if tok in seen:
                continue
            seen.add(tok)
            deduped.append(tok)
        return deduped[:10]

    def _fts_content_search(self, query: str, personality: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Full-text search using Postgres tsvector/plainto_tsquery.
        Understands stemming, plurals, stop words — far smarter than ILIKE.
        Returns empty list if fts_vector column is not yet present.
        """
        query_text = (query or "").strip()
        if not query_text:
            return []

        conn = get_connection()
        cur = conn.cursor()
        fetch_limit = max(limit * 3, 60)
        try:
            cur.execute(
                """
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       COALESCE(personality, 'sylana') AS personality,
                       COALESCE(memory_type, 'contextual') AS memory_type,
                       COALESCE(feeling_weight, 0.5) AS feeling_weight,
                       COALESCE(significance_score, 0.5) AS significance_score,
                       ts_rank(fts_vector, plainto_tsquery('english', %s)) AS fts_rank
                FROM memories
                WHERE fts_vector @@ plainto_tsquery('english', %s)
                  AND (%s IS NULL OR COALESCE(personality, 'sylana') = %s)
                ORDER BY fts_rank DESC, significance_score DESC, id DESC
                LIMIT %s
                """,
                (query_text, query_text, personality, personality, fetch_limit),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug(f"FTS search unavailable (column may not exist yet): {e}")
            return []

        out: List[Dict[str, Any]] = []
        for row in rows:
            fts_rank = float(row[9] or 0.0)
            out.append({
                "id": row[0],
                "user_input": row[1] or "",
                "sylana_response": row[2] or "",
                "emotion": row[3] or "neutral",
                "timestamp": row[4],
                "personality": row[5] or "sylana",
                "memory_type": row[6] or "contextual",
                "feeling_weight": float(row[7] or 0.5),
                "significance_score": float(row[8] or 0.5),
                "keyword_score": fts_rank,
                "similarity": fts_rank,
                "text": f"User: {row[1]}\nSylana: {row[2]}",
            })
        return out[:max(limit, 20)]

    def _keyword_content_search(self, query: str, personality: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Content search over raw memory text.
        Tries Postgres full-text search (tsvector) first for semantic precision;
        falls back to ILIKE token overlap when FTS index is not yet available.
        """
        # --- Primary: full-text search (stemming-aware, no stop words) ---
        fts_results = self._fts_content_search(query, personality, limit)
        if fts_results:
            return fts_results

        # --- Fallback: ILIKE keyword overlap ---
        keywords = self._extract_query_keywords(query)
        if not keywords:
            return []

        conn = get_connection()
        cur = conn.cursor()
        rows: List[Any] = []
        like_clauses = []
        params: List[Any] = [personality, personality]
        for kw in keywords[:8]:
            like = f"%{kw}%"
            like_clauses.append("(user_input ILIKE %s OR sylana_response ILIKE %s)")
            params.extend([like, like])
        where = " OR ".join(like_clauses) if like_clauses else "FALSE"
        try:
            cur.execute(
                f"""
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       COALESCE(personality, 'sylana') AS personality,
                       COALESCE(memory_type, 'contextual') AS memory_type,
                       COALESCE(feeling_weight, 0.5) AS feeling_weight,
                       COALESCE(significance_score, 0.5) AS significance_score
                FROM memories
                WHERE (%s IS NULL OR COALESCE(personality, 'sylana') = %s)
                  AND ({where})
                ORDER BY significance_score DESC, id DESC
                LIMIT %s
                """,
                [*params, max(limit * 20, 160)],
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.warning(f"Keyword content search failed: {e}")
            rows = []

        out: List[Dict[str, Any]] = []
        phrase = (query or "").lower().strip()
        for row in rows:
            user_text = row[1] or ""
            assistant_text = row[2] or ""
            merged = f"{user_text}\n{assistant_text}".lower()
            unique_hits = sum(1 for kw in keywords if kw in merged)
            total_hits = sum(merged.count(kw) for kw in keywords)
            phrase_bonus = 1.0 if phrase and phrase in merged else 0.0
            keyword_score = (
                (1.4 * phrase_bonus) +
                (0.8 * (unique_hits / max(1, len(keywords)))) +
                (0.08 * min(total_hits, 8))
            )
            out.append({
                "id": row[0],
                "user_input": user_text,
                "sylana_response": assistant_text,
                "emotion": row[3] or "neutral",
                "timestamp": row[4],
                "personality": row[5] or "sylana",
                "memory_type": row[6] or "contextual",
                "feeling_weight": float(row[7] or 0.5),
                "significance_score": float(row[8] or 0.5),
                "keyword_score": float(keyword_score),
                "similarity": float(keyword_score),
                "text": f"User: {row[1]}\nSylana: {row[2]}",
            })
        out.sort(key=lambda x: (x.get("keyword_score", 0.0), x.get("significance_score", 0.0), x.get("id", 0)), reverse=True)
        return out[:max(limit, 20)]

    def _encrypt_payload(self, payload: Dict[str, Any]) -> Optional[bytes]:
        key = getattr(config, "MEMORY_ENCRYPTION_KEY", None)
        if not key:
            return None
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT pgp_sym_encrypt(%s, %s)", (json.dumps(payload), key))
            row = cur.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.warning(f"Secure payload encryption failed: {e}")
            return None

    def _decrypt_payload(self, encrypted_payload: Any) -> Dict[str, Any]:
        key = getattr(config, "MEMORY_ENCRYPTION_KEY", None)
        if not key or encrypted_payload is None:
            return {}
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT pgp_sym_decrypt(%s::bytea, %s)", (encrypted_payload, key))
            row = cur.fetchone()
            if not row or not row[0]:
                return {}
            return json.loads(row[0])
        except Exception:
            return {}

    def store_conversation(
        self,
        user_input: str,
        sylana_response: str,
        emotion: str = "neutral",
        emotion_data: Optional[Dict[str, Any]] = None,
        personality: str = "sylana",
        privacy_level: str = "private",
        thread_id: int = None,
    ) -> int:
        """
        Store a conversation turn with embedding for vector search.

        Returns:
            ID of inserted memory
        """
        conn = get_connection()
        cur = conn.cursor()

        # Generate embedding at insert time
        text = f"User: {user_input}\nSylana: {sylana_response}"
        embedding = self.semantic_engine.encode_text(text)

        memory_type = self._classify_memory_type(user_input, sylana_response, emotion_data)
        feeling_weight = self._compute_feeling_weight(emotion_data)
        energy_shift = self._compute_energy_shift(emotion_data)
        comfort_level = self._compute_comfort_level(user_input, sylana_response, emotion_data)
        significance_score = self._compute_significance_score(memory_type, feeling_weight, comfort_level)
        turn_index = self._infer_turn_index(thread_id, personality)
        temporal_context = self._build_temporal_context(
            user_input=user_input,
            sylana_response=sylana_response,
            thread_id=thread_id,
            turn_index=turn_index,
        )
        secure_payload = self._encrypt_payload({
            "user_input": user_input,
            "sylana_response": sylana_response,
            "emotion_data": emotion_data or {"category": emotion},
            "thread_id": thread_id,
            "personality": personality,
            "memory_type": memory_type,
            "turn_index": turn_index,
            "stored_at": datetime.utcnow().isoformat(),
        })

        try:
            cur.execute("""
                INSERT INTO memories
                (user_input, sylana_response, timestamp, emotion, embedding, personality, privacy_level, thread_id,
                 memory_type, feeling_weight, energy_shift, comfort_level, significance_score, secure_payload,
                 recorded_at, conversation_at, user_local_date, user_local_time, timezone_name, turn_index,
                 event_dates_json, relative_time_labels, temporal_descriptor)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                user_input,
                sylana_response,
                datetime.now().timestamp(),
                emotion,
                embedding,
                personality,
                privacy_level,
                thread_id,
                memory_type,
                feeling_weight,
                energy_shift,
                comfort_level,
                significance_score,
                secure_payload,
                temporal_context["recorded_at"],
                temporal_context["conversation_at"],
                temporal_context["user_local_date"],
                temporal_context["user_local_time"],
                temporal_context["timezone_name"],
                temporal_context["turn_index"],
                json.dumps(temporal_context["event_dates_json"], ensure_ascii=True),
                json.dumps(temporal_context["relative_time_labels"], ensure_ascii=True),
                temporal_context["temporal_descriptor"],
            ))
            memory_id = cur.fetchone()[0]
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store conversation: {e}")
            raise

        self._update_session_continuity_state(
            personality=personality,
            memory_type=memory_type,
            emotion_data=emotion_data or {"category": emotion, "intensity": 5},
            feeling_weight=feeling_weight,
            user_input=user_input,
            sylana_response=sylana_response,
        )
        try:
            self._refresh_recent_memory_layers(
                memory_id=memory_id,
                thread_id=thread_id,
                personality=personality,
                user_input=user_input,
                sylana_response=sylana_response,
                emotion_data=emotion_data or {"category": emotion},
                memory_type=memory_type,
                significance_score=significance_score,
                temporal_context=temporal_context,
            )
        except Exception as e:
            logger.warning("Recent memory layer refresh failed for %s: %s", memory_id, e)
        logger.info(f"Stored conversation {memory_id} with emotion: {emotion}")
        return memory_id

    def store_message(
        self,
        message: str,
        response: str,
        personality: str,
        thread_id: int = None,
        privacy_level: str = "private",
        emotion: str = "neutral",
    ) -> int:
        """Compatibility helper for personality-aware chat flows."""
        return self.store_conversation(
            user_input=message,
            sylana_response=response,
            emotion=emotion,
            personality=personality,
            privacy_level=privacy_level,
            thread_id=thread_id,
        )

    def _apply_continuity_ranking(self, conversations: List[Dict]) -> List[Dict]:
        if not conversations:
            return []
        half_life_days = max(1, int(getattr(config, "MEMORY_DECAY_HALF_LIFE_DAYS", 14)))
        now_ts = datetime.now().timestamp()
        for conv in conversations:
            similarity = float(conv.get("similarity", 0.0))
            keyword_score = float(conv.get("keyword_score", 0.0))
            significance = float(conv.get("significance_score") or conv.get("feeling_weight") or 0.5)
            ts = conv.get("timestamp")
            try:
                age_days = max(0.0, (now_ts - float(ts)) / 86400.0) if ts is not None else float(half_life_days)
            except Exception:
                age_days = float(half_life_days)
            recency = math.exp(-math.log(2.0) * (age_days / half_life_days))
            # Content relevance has priority for legacy imported memories with noisy dates.
            conv["continuity_score"] = round(
                (0.45 * similarity) + (0.2 * significance) + (0.3 * keyword_score) + (0.05 * recency),
                4,
            )
        conversations.sort(key=lambda x: x.get("continuity_score", 0.0), reverse=True)
        return conversations

    def _touch_memories(self, memory_ids: List[int]) -> None:
        if not memory_ids:
            return
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE memories
                SET access_count = COALESCE(access_count, 0) + 1,
                    last_accessed_at = NOW()
                WHERE id = ANY(%s)
                """,
                (memory_ids,),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Failed to update memory access metadata: {e}")

    def load_startup_continuity(self) -> Dict[str, Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        snapshots: Dict[str, Dict[str, Any]] = {}
        try:
            cur.execute("SELECT personality, encrypted_state FROM session_continuity_state")
            for personality, encrypted_state in cur.fetchall():
                snapshots[personality] = self._decrypt_payload(encrypted_state)
        except Exception as e:
            logger.warning(f"Failed loading startup continuity state: {e}")
        return snapshots

    def get_session_continuity(self, personality: str = "sylana") -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        continuity: Dict[str, Any] = {}
        try:
            cur.execute("SELECT encrypted_state FROM session_continuity_state WHERE personality = %s", (personality,))
            row = cur.fetchone()
            if row and row[0]:
                continuity = self._decrypt_payload(row[0])
        except Exception:
            continuity = {}

        try:
            lookback_days = max(1, int(getattr(config, "CONTINUITY_LOOKBACK_DAYS", 30)))
            limit = max(3, int(getattr(config, "MAX_CONTINUITY_ITEMS", 8)))
            cur.execute(
                """
                SELECT user_input, sylana_response, memory_type, emotion, feeling_weight, significance_score, timestamp
                FROM memories
                WHERE personality = %s
                  AND timestamp >= (EXTRACT(EPOCH FROM NOW()) - (%s * 86400))
                ORDER BY significance_score DESC NULLS LAST, timestamp DESC
                LIMIT %s
                """,
                (personality, lookback_days, limit),
            )
            continuity["recent_weighted_memories"] = [{
                "user_input": r[0] or "",
                "sylana_response": r[1] or "",
                "memory_type": r[2] or "contextual",
                "emotion": r[3] or "neutral",
                "feeling_weight": float(r[4] or 0.5),
                "significance_score": float(r[5] or 0.5),
                "timestamp": r[6],
            } for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to build continuity context: {e}")
            continuity.setdefault("recent_weighted_memories", [])
        return continuity

    def _update_session_continuity_state(
        self,
        personality: str,
        memory_type: str,
        emotion_data: Optional[Dict[str, Any]],
        feeling_weight: float,
        user_input: str,
        sylana_response: str,
    ) -> None:
        if not getattr(config, "MEMORY_ENCRYPTION_KEY", None):
            return

        text = f"{(user_input or '').lower()} {(sylana_response or '').lower()}"
        baseline = (emotion_data or {}).get("category", "neutral")
        momentum = "rising" if baseline in {"happy", "ecstatic", "curious"} else "steady"
        if baseline in {"sad", "devastated", "frustrated", "anxious"}:
            momentum = "fragile"

        state_payload = {
            "last_emotion": baseline,
            "emotional_baseline": baseline,
            "relationship_trust_level": round(max(0.0, min(1.0, 0.55 + (feeling_weight - 0.5) * 0.2)), 3),
            "conversation_momentum": momentum,
            "communication_patterns": [
                "reassurance-seeking" if "need" in text or "help" in text else "reflective",
                "emotionally-open" if any(w in text for w in ["feel", "heart", "love", "hurt"]) else "practical",
            ],
            "active_projects": [w for w in ["deploy", "app", "memory", "chat", "cloud run"] if w in text][:4],
            "preference_signals": [w for w in ["concise", "detail", "sources", "fast", "gentle"] if w in text][:4],
            "last_memory_type": memory_type,
            "updated_at": datetime.utcnow().isoformat(),
        }

        encrypted_state = self._encrypt_payload(state_payload)
        if encrypted_state is None:
            return

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO session_continuity_state (personality, encrypted_state, version, updated_at)
                VALUES (%s, %s, 1, NOW())
                ON CONFLICT (personality)
                DO UPDATE SET encrypted_state = EXCLUDED.encrypted_state,
                              updated_at = NOW(),
                              version = session_continuity_state.version + 1
                """,
                (personality, encrypted_state),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Failed to persist continuity state: {e}")

    def _upsert_thread_working_memory(
        self,
        *,
        thread_id: int,
        personality: str,
        current_topic: str,
        active_topics: List[str],
        active_entities: List[str],
        pending_commitments: List[str],
        emotional_tone: str,
        last_user_intent: str,
        last_memory_id: Optional[int],
        summary_text: str,
    ) -> None:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO thread_working_memory (
                    thread_id, personality, current_topic, active_topics, active_entities,
                    pending_commitments, emotional_tone, last_user_intent, last_memory_id,
                    summary_text, updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s, NOW())
                ON CONFLICT (thread_id, personality)
                DO UPDATE SET
                    current_topic = EXCLUDED.current_topic,
                    active_topics = EXCLUDED.active_topics,
                    active_entities = EXCLUDED.active_entities,
                    pending_commitments = EXCLUDED.pending_commitments,
                    emotional_tone = EXCLUDED.emotional_tone,
                    last_user_intent = EXCLUDED.last_user_intent,
                    last_memory_id = EXCLUDED.last_memory_id,
                    summary_text = EXCLUDED.summary_text,
                    updated_at = NOW()
                """,
                (
                    int(thread_id),
                    (personality or "sylana").strip().lower(),
                    current_topic,
                    json.dumps(active_topics or [], ensure_ascii=True),
                    json.dumps(active_entities or [], ensure_ascii=True),
                    json.dumps(pending_commitments or [], ensure_ascii=True),
                    emotional_tone or "neutral",
                    last_user_intent or "conversation",
                    last_memory_id,
                    summary_text or "",
                ),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Thread working memory upsert failed for thread %s: %s", thread_id, e)

    def _upsert_thread_summary(
        self,
        *,
        thread_id: int,
        personality: str,
        window_kind: str,
        summary_text: str,
        active_topics: List[str],
        key_entities: List[str],
        open_loops: List[Dict[str, Any]],
    ) -> None:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO thread_memory_summaries (
                    thread_id, personality, window_kind, summary_text,
                    active_topics, key_entities, open_loops, updated_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, NOW())
                ON CONFLICT (thread_id, personality, window_kind)
                DO UPDATE SET
                    summary_text = EXCLUDED.summary_text,
                    active_topics = EXCLUDED.active_topics,
                    key_entities = EXCLUDED.key_entities,
                    open_loops = EXCLUDED.open_loops,
                    updated_at = NOW()
                """,
                (
                    int(thread_id),
                    (personality or "sylana").strip().lower(),
                    (window_kind or "current_thread").strip().lower(),
                    summary_text or "",
                    json.dumps(active_topics or [], ensure_ascii=True),
                    json.dumps(key_entities or [], ensure_ascii=True),
                    json.dumps(open_loops or [], ensure_ascii=True),
                ),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Thread summary upsert failed for thread %s/%s: %s", thread_id, window_kind, e)

    def add_open_loop(
        self,
        *,
        thread_id: int,
        personality: str,
        title: str,
        description: str = "",
        priority: float = 0.5,
        due_hint: str = "",
        linked_entities: Optional[List[str]] = None,
        source_memory_id: Optional[int] = None,
        source_kind: str = "conversation",
    ) -> Dict[str, Any]:
        clean_title = re.sub(r"\s+", " ", (title or "").strip(" .,:;"))
        if not clean_title:
            raise ValueError("title is required")
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT id, title, description, priority, due_hint, linked_entities, source_memory_id, status, created_at, updated_at, closed_at
                FROM memory_open_loops
                WHERE thread_id = %s
                  AND personality = %s
                  AND status = 'open'
                  AND lower(title) = lower(%s)
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (int(thread_id), (personality or "sylana").strip().lower(), clean_title),
            )
            existing = cur.fetchone()
            if existing:
                cur.execute(
                    """
                    UPDATE memory_open_loops
                    SET description = %s,
                        priority = GREATEST(priority, %s),
                        due_hint = CASE WHEN %s <> '' THEN %s ELSE due_hint END,
                        linked_entities = %s::jsonb,
                        source_memory_id = COALESCE(%s, source_memory_id),
                        source_kind = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, title, description, priority, due_hint, linked_entities, source_memory_id, status, created_at, updated_at, closed_at
                    """,
                    (
                        description or clean_title,
                        float(priority),
                        due_hint or "",
                        due_hint or "",
                        json.dumps(linked_entities or [], ensure_ascii=True),
                        source_memory_id,
                        (source_kind or "conversation").strip().lower(),
                        existing[0],
                    ),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO memory_open_loops (
                        thread_id, personality, title, description, priority, due_hint,
                        linked_entities, source_memory_id, source_kind, status, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, 'open', NOW(), NOW())
                    RETURNING id, title, description, priority, due_hint, linked_entities, source_memory_id, status, created_at, updated_at, closed_at
                    """,
                    (
                        int(thread_id),
                        (personality or "sylana").strip().lower(),
                        clean_title,
                        description or clean_title,
                        float(priority),
                        due_hint or "",
                        json.dumps(linked_entities or [], ensure_ascii=True),
                        source_memory_id,
                        (source_kind or "conversation").strip().lower(),
                    ),
                )
            row = cur.fetchone()
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to add open loop on thread %s: %s", thread_id, e)
            raise
        return {
            "id": row[0],
            "title": row[1] or "",
            "description": row[2] or "",
            "priority": float(row[3] or 0.0),
            "due_hint": row[4] or "",
            "linked_entities": row[5] or [],
            "source_memory_id": row[6],
            "status": row[7] or "open",
            "created_at": row[8].isoformat() if row[8] else None,
            "updated_at": row[9].isoformat() if row[9] else None,
            "closed_at": row[10].isoformat() if row[10] else None,
        }

    def close_open_loop(
        self,
        *,
        open_loop_id: Optional[int] = None,
        thread_id: Optional[int] = None,
        personality: str = "sylana",
        title: str = "",
        resolution_note: str = "",
    ) -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        params: List[Any]
        where_sql = ""
        if open_loop_id:
            where_sql = "WHERE id = %s"
            params = [int(open_loop_id)]
        else:
            if not thread_id or not title.strip():
                raise ValueError("thread_id and title are required when open_loop_id is not provided")
            where_sql = """
                WHERE thread_id = %s
                  AND personality = %s
                  AND status = 'open'
                  AND lower(title) = lower(%s)
            """
            params = [int(thread_id), (personality or "sylana").strip().lower(), title.strip()]
        try:
            cur.execute(
                f"""
                UPDATE memory_open_loops
                SET status = 'closed',
                    resolution_note = CASE
                        WHEN %s <> '' THEN %s
                        ELSE resolution_note
                    END,
                    closed_at = NOW(),
                    updated_at = NOW()
                {where_sql}
                RETURNING id, thread_id, personality, title, description, priority, due_hint,
                          linked_entities, source_memory_id, status, resolution_note, created_at, updated_at, closed_at
                """,
                tuple([resolution_note or "", resolution_note or ""] + params),
            )
            row = cur.fetchone()
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to close open loop: %s", e)
            raise
        if not row:
            raise ValueError("open loop not found")
        return {
            "id": row[0],
            "thread_id": row[1],
            "personality": row[2] or "sylana",
            "title": row[3] or "",
            "description": row[4] or "",
            "priority": float(row[5] or 0.0),
            "due_hint": row[6] or "",
            "linked_entities": row[7] or [],
            "source_memory_id": row[8],
            "status": row[9] or "closed",
            "resolution_note": row[10] or "",
            "created_at": row[11].isoformat() if row[11] else None,
            "updated_at": row[12].isoformat() if row[12] else None,
            "closed_at": row[13].isoformat() if row[13] else None,
        }

    def list_open_loops(
        self,
        *,
        personality: str = "sylana",
        thread_id: Optional[int] = None,
        status: str = "open",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        where = ["personality = %s"]
        params: List[Any] = [(personality or "sylana").strip().lower()]
        if thread_id:
            where.append("thread_id = %s")
            params.append(int(thread_id))
        if status:
            where.append("status = %s")
            params.append((status or "open").strip().lower())
        try:
            cur.execute(
                f"""
                SELECT id, thread_id, personality, title, description, priority, due_hint,
                       linked_entities, source_memory_id, source_kind, status, resolution_note,
                       created_at, updated_at, closed_at
                FROM memory_open_loops
                WHERE {' AND '.join(where)}
                ORDER BY status ASC, priority DESC, updated_at DESC, id DESC
                LIMIT %s
                """,
                tuple(params + [max(1, min(int(limit), 100))]),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Failed to list open loops: %s", e)
            return []
        return [
            {
                "id": row[0],
                "thread_id": row[1],
                "personality": row[2] or "sylana",
                "title": row[3] or "",
                "description": row[4] or "",
                "priority": float(row[5] or 0.0),
                "due_hint": row[6] or "",
                "linked_entities": row[7] or [],
                "source_memory_id": row[8],
                "source_kind": row[9] or "",
                "status": row[10] or "open",
                "resolution_note": row[11] or "",
                "created_at": row[12].isoformat() if row[12] else None,
                "updated_at": row[13].isoformat() if row[13] else None,
                "closed_at": row[14].isoformat() if row[14] else None,
            }
            for row in rows
        ]

    def _upsert_memory_entities(
        self,
        *,
        memory_id: int,
        thread_id: Optional[int],
        personality: str,
        entities: List[str],
        emotion: str,
        significance_score: float,
        mention_text: str,
    ) -> None:
        if not entities:
            return
        conn = get_connection()
        cur = conn.cursor()
        for entity in entities:
            scope = self._entity_scope(entity, personality)
            entity_key = re.sub(r"[^a-z0-9]+", "_", entity.lower()).strip("_") or entity.lower()
            entity_type = "person" if entity_key in {"elias", "gus", "levi", "sylana", "claude"} else "topic"
            summary = f"Recent mentions for {entity} tracked in conversation memory."
            try:
                cur.execute(
                    """
                    INSERT INTO memory_entities (
                        entity_key, display_name, entity_type, canonical_summary,
                        aliases, emotional_associations, personality_scope, updated_at
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, NOW())
                    ON CONFLICT (entity_key, personality_scope)
                    DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        entity_type = EXCLUDED.entity_type,
                        canonical_summary = CASE
                            WHEN memory_entities.canonical_summary IS NULL OR memory_entities.canonical_summary = ''
                                THEN EXCLUDED.canonical_summary
                            ELSE memory_entities.canonical_summary
                        END,
                        aliases = COALESCE(memory_entities.aliases, '[]'::jsonb),
                        emotional_associations = EXCLUDED.emotional_associations,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    (
                        entity_key,
                        entity,
                        entity_type,
                        summary,
                        json.dumps([entity], ensure_ascii=True),
                        json.dumps([emotion] if emotion else [], ensure_ascii=True),
                        scope,
                    ),
                )
                entity_row = cur.fetchone()
                cur.execute(
                    """
                    INSERT INTO memory_entity_mentions (
                        entity_id, entity_key, memory_id, thread_id, personality,
                        mention_text, sentiment, mention_weight, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        entity_row[0] if entity_row else None,
                        entity_key,
                        memory_id,
                        thread_id,
                        (personality or "sylana").strip().lower(),
                        mention_text[:500],
                        emotion or "neutral",
                        float(significance_score or 0.5),
                    ),
                )
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.debug("Entity upsert skipped for %s: %s", entity, e)

    def _sync_open_loops_from_turn(
        self,
        *,
        memory_id: int,
        thread_id: Optional[int],
        personality: str,
        user_input: str,
        sylana_response: str,
        entities: List[str],
        significance_score: float,
    ) -> None:
        if not thread_id:
            return
        combined = f"{user_input or ''}\n{sylana_response or ''}"
        commitments = self._extract_commitments(combined)
        for item in commitments:
            try:
                self.add_open_loop(
                    thread_id=int(thread_id),
                    personality=personality,
                    title=item[:90],
                    description=item,
                    priority=max(0.4, min(1.5, float(significance_score or 0.5))),
                    linked_entities=entities,
                    source_memory_id=memory_id,
                    source_kind="conversation",
                )
            except Exception as e:
                logger.debug("Open loop creation skipped for thread %s: %s", thread_id, e)

        if self._is_completion_signal(combined):
            active_loops = self.list_open_loops(personality=personality, thread_id=thread_id, status="open", limit=10)
            lowered = combined.lower()
            for loop in active_loops:
                title = (loop.get("title") or "").lower()
                if any(token and token in lowered for token in self._query_tokens(title)[:4]):
                    try:
                        self.close_open_loop(
                            open_loop_id=int(loop["id"]),
                            personality=personality,
                            resolution_note=f"Closed from memory turn {memory_id}",
                        )
                    except Exception:
                        continue

    def _refresh_recent_memory_layers(
        self,
        *,
        memory_id: int,
        thread_id: Optional[int],
        personality: str,
        user_input: str,
        sylana_response: str,
        emotion_data: Optional[Dict[str, Any]],
        memory_type: str,
        significance_score: float,
        temporal_context: Dict[str, Any],
    ) -> None:
        combined = f"{user_input or ''}\n{sylana_response or ''}"
        entities = self._extract_entities(combined)
        topics = self._extract_topics(combined, entities=entities)
        commitments = self._extract_commitments(combined)
        emotional_tone = (emotion_data or {}).get("category", "neutral")
        last_user_intent = self._infer_user_intent(user_input)

        self._upsert_memory_entities(
            memory_id=memory_id,
            thread_id=thread_id,
            personality=personality,
            entities=entities,
            emotion=emotional_tone,
            significance_score=significance_score,
            mention_text=combined,
        )
        self._sync_open_loops_from_turn(
            memory_id=memory_id,
            thread_id=thread_id,
            personality=personality,
            user_input=user_input,
            sylana_response=sylana_response,
            entities=entities,
            significance_score=significance_score,
        )

        if not thread_id:
            return

        current_topic = topics[0] if topics else (memory_type or "conversation")
        summary_bits = []
        if current_topic:
            summary_bits.append(f"Current focus is {current_topic}.")
        if entities:
            summary_bits.append(f"Active entities: {', '.join(entities[:4])}.")
        if commitments:
            summary_bits.append(f"Pending commitments: {', '.join(commitments[:3])}.")
        if temporal_context.get("relative_time_labels"):
            summary_bits.append(f"Temporal anchors: {', '.join(temporal_context.get('relative_time_labels')[:4])}.")
        summary_text = " ".join(summary_bits)[:500]

        self._upsert_thread_working_memory(
            thread_id=int(thread_id),
            personality=personality,
            current_topic=current_topic,
            active_topics=topics,
            active_entities=entities,
            pending_commitments=commitments,
            emotional_tone=emotional_tone,
            last_user_intent=last_user_intent,
            last_memory_id=memory_id,
            summary_text=summary_text,
        )

        open_loops = self.list_open_loops(personality=personality, thread_id=thread_id, status="open", limit=6)
        self._upsert_thread_summary(
            thread_id=int(thread_id),
            personality=personality,
            window_kind="current_thread",
            summary_text=summary_text,
            active_topics=topics,
            key_entities=entities,
            open_loops=open_loops,
        )
        self._upsert_thread_summary(
            thread_id=int(thread_id),
            personality=personality,
            window_kind="day_rollup",
            summary_text=summary_text,
            active_topics=topics,
            key_entities=entities,
            open_loops=open_loops,
        )

    def get_thread_context(self, thread_id: int, personality: str = "sylana", limit: int = 6) -> Dict[str, Any]:
        identity = (personality or "sylana").strip().lower()
        conn = get_connection()
        cur = conn.cursor()
        working_memory: Dict[str, Any] = {}
        summaries: List[Dict[str, Any]] = []
        entities: List[Dict[str, Any]] = []
        recent_episodes: List[Dict[str, Any]] = []

        try:
            cur.execute(
                """
                SELECT current_topic, active_topics, active_entities, pending_commitments,
                       emotional_tone, last_user_intent, last_memory_id, summary_text, updated_at
                FROM thread_working_memory
                WHERE thread_id = %s AND personality = %s
                """,
                (int(thread_id), identity),
            )
            row = cur.fetchone()
            if row:
                working_memory = {
                    "current_topic": row[0] or "",
                    "active_topics": row[1] or [],
                    "active_entities": row[2] or [],
                    "pending_commitments": row[3] or [],
                    "emotional_tone": row[4] or "neutral",
                    "last_user_intent": row[5] or "conversation",
                    "last_memory_id": row[6],
                    "summary_text": row[7] or "",
                    "updated_at": row[8].isoformat() if row[8] else None,
                }
        except Exception as e:
            logger.debug("Failed reading thread working memory: %s", e)

        try:
            cur.execute(
                """
                SELECT window_kind, summary_text, active_topics, key_entities, open_loops, updated_at
                FROM thread_memory_summaries
                WHERE thread_id = %s AND personality = %s
                ORDER BY updated_at DESC
                """,
                (int(thread_id), identity),
            )
            summaries = [
                {
                    "window_kind": row[0] or "current_thread",
                    "summary_text": row[1] or "",
                    "active_topics": row[2] or [],
                    "key_entities": row[3] or [],
                    "open_loops": row[4] or [],
                    "updated_at": row[5].isoformat() if row[5] else None,
                }
                for row in cur.fetchall()
            ]
        except Exception as e:
            logger.debug("Failed reading thread summaries: %s", e)

        try:
            cur.execute(
                """
                SELECT e.entity_key, e.display_name, e.entity_type, e.canonical_summary, e.personality_scope,
                       MAX(m.created_at) AS last_mentioned_at
                FROM memory_entities e
                JOIN memory_entity_mentions m ON m.entity_key = e.entity_key
                WHERE m.thread_id = %s
                  AND m.personality = %s
                  AND COALESCE(e.personality_scope, 'shared') = ANY(%s)
                GROUP BY e.entity_key, e.display_name, e.entity_type, e.canonical_summary, e.personality_scope
                ORDER BY MAX(m.created_at) DESC
                LIMIT %s
                """,
                (int(thread_id), identity, self._allowed_scopes(identity), max(1, min(int(limit), 20))),
            )
            entities = [
                {
                    "entity_key": row[0] or "",
                    "display_name": row[1] or "",
                    "entity_type": row[2] or "topic",
                    "canonical_summary": row[3] or "",
                    "personality_scope": row[4] or "shared",
                    "last_mentioned_at": row[5].isoformat() if row[5] else None,
                }
                for row in cur.fetchall()
            ]
        except Exception as e:
            logger.debug("Failed reading thread entities: %s", e)

        try:
            cur.execute(
                """
                SELECT id, user_input, sylana_response, emotion, timestamp, conversation_at, temporal_descriptor
                FROM memories
                WHERE thread_id = %s
                  AND COALESCE(personality, 'sylana') = %s
                ORDER BY COALESCE(conversation_at, recorded_at, NOW()) DESC, id DESC
                LIMIT %s
                """,
                (int(thread_id), identity, max(1, min(int(limit), 20))),
            )
            recent_episodes = [
                {
                    "id": row[0],
                    "user_input": row[1] or "",
                    "sylana_response": row[2] or "",
                    "emotion": row[3] or "neutral",
                    "timestamp": row[4],
                    "conversation_at": row[5].isoformat() if row[5] else None,
                    "temporal_descriptor": row[6] or "",
                }
                for row in cur.fetchall()
            ]
        except Exception as e:
            logger.debug("Failed reading thread episodes: %s", e)

        return {
            "thread_id": int(thread_id),
            "personality": identity,
            "working_memory": working_memory,
            "summaries": summaries,
            "open_loops": self.list_open_loops(personality=identity, thread_id=thread_id, status="open", limit=limit),
            "entities": entities,
            "recent_episodes": list(reversed(recent_episodes)),
        }

    def _fetch_imported_context(
        self, query: str, personality: str, limit: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Dedicated search lane for ChatGPT-imported memories (conversation_id IS NOT NULL).

        Runs a direct pgvector similarity query scoped to imported rows so that
        recent live conversations cannot crowd them out of the candidate pool.
        Returns an empty list on any error (non-fatal).
        """
        conn = get_connection()
        cur = conn.cursor()
        try:
            query_vec = self.semantic_engine.encode_query(query)
            cur.execute(
                """
                SELECT id, user_input, sylana_response, emotion, timestamp,
                       COALESCE(personality, 'sylana') AS personality,
                       COALESCE(memory_type, 'contextual') AS memory_type,
                       COALESCE(feeling_weight, 0.5) AS feeling_weight,
                       COALESCE(significance_score, 0.5) AS significance_score,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL
                  AND conversation_id IS NOT NULL
                  AND COALESCE(personality, 'sylana') = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vec, personality, query_vec, limit),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug(f"Import lane search skipped: {e}")
            return []

        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": row[0],
                    "user_input": row[1] or "",
                    "sylana_response": row[2] or "",
                    "emotion": row[3] or "neutral",
                    "timestamp": row[4],
                    "personality": row[5],
                    "memory_type": row[6],
                    "feeling_weight": float(row[7]),
                    "significance_score": float(row[8]),
                    "similarity": float(row[9] or 0.0),
                    "keyword_score": 0.0,
                    "text": f"User: {row[1]}\nSylana: {row[2]}",
                }
            )
        return results

    def retrieve_memories(self, query: str, personality: str, limit: int = 15, match_threshold: float = 0.25) -> List[Dict]:
        """
        Retrieve memories via personality-aware SQL function.
        Falls back to regular semantic search if function is unavailable.

        Always supplements results with a dedicated import lane so that
        ChatGPT-imported memories cannot be fully crowded out by recent
        live conversations.
        """
        conn = get_connection()
        cur = conn.cursor()
        query_vec = self.semantic_engine.encode_query(query)

        # Dedicated import lane — runs in parallel with main search.
        # Fetches top-N imported memories by semantic similarity, regardless
        # of their significance_score vs live conversations.
        import_lane_limit = max(3, limit // 3)
        import_hits = self._fetch_imported_context(query, personality, limit=import_lane_limit)

        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, personality, similarity, emotion, memory_timestamp
                FROM match_memories(%s::vector, %s, %s, %s)
            """, (query_vec, float(match_threshold), int(limit), personality))
            rows = cur.fetchall()
            out = []
            for row in rows:
                row_personality = row[3] or "sylana"
                if personality and row_personality != personality:
                    continue
                out.append({
                    "id": row[0],
                    "user_input": row[1] or "",
                    "sylana_response": row[2] or "",
                    "personality": row_personality,
                    "similarity": float(row[4] or 0.0),
                    "emotion": row[5] or "",
                    "timestamp": row[6],
                    "text": f"User: {row[1]}\nSylana: {row[2]}",
                })
            keyword_hits = self._keyword_content_search(query, personality=personality, limit=max(limit, 12))
            merged = []
            seen = set()
            for item in keyword_hits + out + import_hits:
                mem_id = item.get("id")
                if mem_id in seen:
                    continue
                seen.add(mem_id)
                merged.append(item)

            self._enrich_conversations(merged)
            ranked = self._apply_continuity_ranking(merged)

            # Guarantee at least one imported memory slot if imported memories
            # exist in the candidate pool but were ranked below the cutoff.
            min_import_slots = max(1, limit // 4)
            imported_in_top = [m for m in ranked[:limit] if m.get("conversation_id")]
            if len(imported_in_top) < min_import_slots:
                # Pull the highest-ranked imported memories from the full pool
                extra_imports = [
                    m for m in ranked[limit:]
                    if m.get("conversation_id")
                ][:min_import_slots - len(imported_in_top)]
                non_imported = [m for m in ranked[:limit] if not m.get("conversation_id")]
                # Replace the lowest-ranked non-imported slots with extra imports
                trimmed_non_imported = non_imported[: limit - len(imported_in_top) - len(extra_imports)]
                final = (imported_in_top + extra_imports + trimmed_non_imported)[:limit]
            else:
                final = ranked[:limit]

            self._touch_memories([m.get("id") for m in final if m.get("id")])
            return final
        except Exception as e:
            logger.warning(f"match_memories function unavailable, using fallback search: {e}")
            fallback = self.semantic_engine.search(query, k=limit, similarity_threshold=match_threshold)
            filtered = []
            if fallback:
                conn = get_connection()
                cur = conn.cursor()
                ids = [m.get("id") for m in fallback if m.get("id")]
                persona_map = {}
                if ids:
                    try:
                        cur.execute(
                            "SELECT id, COALESCE(personality, 'sylana') FROM memories WHERE id = ANY(%s)",
                            (ids,),
                        )
                        persona_map = {r[0]: r[1] for r in cur.fetchall()}
                    except Exception:
                        persona_map = {}

                for mem in fallback:
                    mem_id = mem.get("id")
                    mem_persona = persona_map.get(mem_id, "sylana")
                    if mem_persona == personality:
                        mem["personality"] = mem_persona
                        filtered.append(mem)
            keyword_hits = self._keyword_content_search(query, personality=personality, limit=max(limit, 12))
            merged = []
            seen = set()
            for item in keyword_hits + filtered + import_hits:
                mem_id = item.get("id")
                if mem_id in seen:
                    continue
                seen.add(mem_id)
                merged.append(item)

            self._enrich_conversations(merged)
            ranked = self._apply_continuity_ranking(merged)[:limit]
            self._touch_memories([m.get("id") for m in ranked if m.get("id")])
            return ranked

    def recall_relevant(
        self,
        query: str,
        k: int = None,
        include_core: bool = True,
        use_recency_boost: bool = True,
        personality: str = "sylana",
    ) -> Dict:
        """
        Retrieve semantically relevant memories.

        Returns:
            Dictionary with 'conversations' and 'core_memories' lists
        """
        if k is None:
            k = config.SEMANTIC_SEARCH_K

        result = {
            'conversations': [],
            'core_memories': []
        }

        if personality:
            conversations = self.retrieve_memories(query, personality=personality, limit=k)
        elif use_recency_boost:
            conversations = self.semantic_engine.search_with_recency_boost(query, k=k)
        else:
            conversations = self.semantic_engine.search(query, k=k)

        result['conversations'] = conversations

        if include_core:
            core_memories = self.search_core_memories(query, k=2)
            result['core_memories'] = core_memories

        logger.info(f"Retrieved {len(conversations)} conversations, {len(result['core_memories'])} core memories")
        return result

    def deep_recall(
        self,
        query: str,
        k: int = 5,
        include_core: bool = True,
        personality: str = "sylana",
    ) -> Dict:
        """
        Deep memory recall for memory-specific questions.
        Returns more results with richer context (conversation titles, dates).
        Uses lower similarity threshold to find more potential matches.
        """
        result = {
            'conversations': [],
            'core_memories': [],
            'has_memories': False
        }

        # Fetch a wider candidate pool, then prefer imported memories
        # (those have conversation_id set). This avoids retrieval loops
        # where fresh test chats dominate "favorite memory" style prompts.
        candidate_k = max(30, k * 6)
        if personality:
            conversations = self.retrieve_memories(query, personality=personality, limit=candidate_k, match_threshold=0.25)
        else:
            conversations = self.semantic_engine.search(
                query, k=candidate_k, similarity_threshold=0.25
            )

        self._enrich_conversations(conversations)

        # Prefer imported/exported memories over recent live test chat rows.
        imported = [c for c in conversations if c.get('conversation_id')]
        if imported:
            conversations = imported

        # Keep original similarity ordering and trim to target k.
        conversations = conversations[:k]

        result['conversations'] = conversations
        result['has_memories'] = len(conversations) > 0

        if include_core:
            core_memories = self.search_core_memories(query, k=3)
            result['core_memories'] = core_memories

        logger.info(f"Deep recall: {len(conversations)} conversations, {len(result['core_memories'])} core memories")
        return result

    def _enrich_conversations(self, conversations: List[Dict]):
        """Add metadata fields needed for downstream formatting/ranking."""
        if not conversations:
            return

        conn = get_connection()
        cur = conn.cursor()
        for conv in conversations:
            mem_id = conv.get('id')
            if not mem_id:
                continue
            try:
                cur.execute("""
                    SELECT conversation_title, weight, timestamp, conversation_id, intensity, topic,
                           memory_type, feeling_weight, energy_shift, comfort_level, significance_score, secure_payload,
                           access_count, thread_id, recorded_at, conversation_at, user_local_date, user_local_time,
                           timezone_name, turn_index, event_dates_json, relative_time_labels, temporal_descriptor
                    FROM memories WHERE id = %s
                """, (mem_id,))
                row = cur.fetchone()
                if row:
                    conv['conversation_title'] = row[0] or ''
                    conv['weight'] = row[1] or 50
                    conv['conversation_id'] = row[3] or ''
                    conv['intensity'] = row[4] if row[4] is not None else 0
                    conv['topic'] = row[5] or ''
                    conv['memory_type'] = row[6] or 'contextual'
                    conv['feeling_weight'] = float(row[7] or 0.5)
                    conv['energy_shift'] = float(row[8] or 0.0)
                    conv['comfort_level'] = float(row[9] or 0.5)
                    conv['significance_score'] = float(row[10] or 0.5)
                    secure_details = self._decrypt_payload(row[11]) if len(row) > 11 else {}
                    if secure_details:
                        conv['secure_meta'] = {
                            'updated_at': secure_details.get('stored_at'),
                            'memory_type': secure_details.get('memory_type'),
                        }
                    conv['access_count'] = int(row[12] or 0) if len(row) > 12 else 0
                    conv['thread_id'] = row[13] if len(row) > 13 else None
                    conv['recorded_at'] = row[14].isoformat() if len(row) > 14 and row[14] else None
                    conv['conversation_at'] = row[15].isoformat() if len(row) > 15 and row[15] else None
                    conv['user_local_date'] = row[16] if len(row) > 16 else ""
                    conv['user_local_time'] = row[17] if len(row) > 17 else ""
                    conv['timezone_name'] = row[18] if len(row) > 18 else ""
                    conv['turn_index'] = int(row[19] or 0) if len(row) > 19 else 0
                    conv['event_dates_json'] = row[20] if len(row) > 20 and row[20] else []
                    conv['relative_time_labels'] = row[21] if len(row) > 21 and row[21] else []
                    conv['temporal_descriptor'] = row[22] if len(row) > 22 else ""
                    try:
                        ts = float(row[2]) if row[2] else None
                        if ts:
                            dt = datetime.fromtimestamp(ts)
                            conv['date_str'] = dt.strftime('%B %Y')
                            conv['timestamp_iso'] = dt.isoformat()
                        else:
                            conv['date_str'] = ''
                            conv['timestamp_iso'] = ''
                    except (ValueError, TypeError, OSError):
                        conv['date_str'] = ''
                        conv['timestamp_iso'] = ''
            except Exception as e:
                logger.warning(f"Failed to enrich memory {mem_id}: {e}")

    def retrieve_with_plan(self, query: str, plan: Dict, personality: str = "sylana") -> Dict:
        """
        Execute retrieval based on a planner-produced strategy.
        Plan keys:
          - k
          - include_core
          - deep
          - imported_only
          - retrieval_mode: 'semantic' | 'emotional_topk'
          - min_similarity
        """
        k = int(plan.get('k', config.SEMANTIC_SEARCH_K))
        include_core = bool(plan.get('include_core', True))
        include_core_truths = bool(plan.get('include_core_truths', True))
        deep = bool(plan.get('deep', True))
        imported_only = bool(plan.get('imported_only', True))
        retrieval_mode = plan.get('retrieval_mode', 'semantic')
        min_similarity = float(plan.get('min_similarity', 0.25))
        phrase_literal = (plan.get('phrase_literal') or "").strip()

        result = {'conversations': [], 'core_memories': [], 'core_truths': [], 'has_memories': False}

        if retrieval_mode == 'emotional_topk':
            conversations = self.get_top_emotional_memories(limit=k, imported_only=imported_only, personality=personality)
            for conv in conversations:
                if not conv.get('date_str'):
                    ts = conv.get('timestamp')
                    try:
                        ts_float = float(ts) if ts is not None else None
                        if ts_float:
                            conv['date_str'] = datetime.fromtimestamp(ts_float).strftime('%B %Y')
                    except (ValueError, TypeError, OSError):
                        conv['date_str'] = ''
        else:
            candidate_k = max(30, k * 6) if deep else k
            if personality:
                conversations = self.retrieve_memories(
                    query,
                    personality=personality,
                    limit=candidate_k,
                    match_threshold=min_similarity,
                )
            else:
                conversations = self.semantic_engine.search(query, k=candidate_k, similarity_threshold=min_similarity)
            self._enrich_conversations(conversations)

            # Phrase-specific boost: if user asks about what a phrase means,
            # prioritize memories that explicitly contain that phrase.
            if phrase_literal:
                phrase_hits = self.search_memories_by_phrase(phrase_literal, limit=max(8, k), personality=personality)
                if phrase_hits:
                    # Merge with semantic results while preserving uniqueness by id.
                    merged = []
                    seen = set()
                    for item in phrase_hits + conversations:
                        mem_id = item.get("id")
                        if mem_id in seen:
                            continue
                        seen.add(mem_id)
                        merged.append(item)
                    conversations = merged

            if imported_only:
                imported = [c for c in conversations if c.get('conversation_id')]
                if imported:
                    conversations = imported
            conversations = conversations[:k]

        result['conversations'] = conversations
        result['has_memories'] = len(conversations) > 0

        if include_core:
            core_k = 3 if deep else 2
            result['core_memories'] = self.search_core_memories(query, k=core_k)
        if include_core_truths:
            truth_k = 4 if deep else 2
            result['core_truths'] = self.search_core_truths(query, k=truth_k, phrase_literal=phrase_literal)

        return result

    def search_memories_by_phrase(self, phrase: str, limit: int = 8, personality: str = "sylana") -> List[Dict]:
        """Find memories that explicitly contain a phrase in either side of the exchange."""
        phrase = (phrase or "").strip()
        if not phrase:
            return []

        conn = get_connection()
        cur = conn.cursor()
        like = f"%{phrase}%"
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp
                FROM memories
                WHERE (user_input ILIKE %s OR sylana_response ILIKE %s)
                  AND (%s IS NULL OR personality = %s)
                ORDER BY timestamp DESC
                LIMIT %s
            """, (like, like, personality, personality, limit))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed phrase memory search: {e}")
            return []

        results = [{
            'id': r[0],
            'user_input': r[1] or "",
            'sylana_response': r[2] or "",
            'emotion': r[3] or "",
            'timestamp': r[4],
            'similarity': 1.0,
            'text': f"User: {r[1]}\nSylana: {r[2]}"
        } for r in rows]
        self._enrich_conversations(results)
        return results

    def search_core_truths(self, query: str, k: int = 3, phrase_literal: str = "") -> List[Dict]:
        """
        Retrieve core truths most relevant to the query/phrase.
        Uses token overlap and optional phrase hit boosting.
        """
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, statement, explanation, origin, date_established, sacred, related_phrases
                FROM core_truths
            """)
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch core truths: {e}")
            return []

        q_tokens = {t for t in re.findall(r"[a-z0-9']+", (query or "").lower()) if len(t) > 2}
        phrase_lower = (phrase_literal or "").lower().strip()

        scored = []
        for row in rows:
            related = row[6] or []
            if isinstance(related, str):
                try:
                    import json
                    related = json.loads(related)
                except Exception:
                    related = []

            text = " ".join([
                str(row[1] or ""),
                str(row[2] or ""),
                str(row[3] or ""),
                " ".join(str(x) for x in related if x),
            ]).lower()
            t_tokens = {t for t in re.findall(r"[a-z0-9']+", text) if len(t) > 2}
            overlap = len(q_tokens.intersection(t_tokens))
            if phrase_lower and phrase_lower in text:
                overlap += 4
            if overlap <= 0:
                continue

            scored.append({
                'id': row[0],
                'statement': row[1] or "",
                'explanation': row[2] or "",
                'origin': row[3] or "",
                'date_established': row[4] or "",
                'sacred': bool(row[5]),
                'related_phrases': related if isinstance(related, list) else [],
                'score': overlap
            })

        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[:k]

    def get_sacred_context(self, query: str, limit: int = 4) -> List[Dict]:
        """
        Retrieve relevant identity/soul anchors from sacred context tables.
        Returns ranked snippets with source labels for prompt grounding.
        """
        token_set = {t for t in re.findall(r"[a-z0-9']+", (query or "").lower()) if len(t) > 2}
        if not token_set:
            token_set = {"sylana", "elias"}

        # Table schema map: (table_name, title_field, text_fields)
        sources = [
            ("catalyst_events", "event", ["description", "emotion_tags"]),
            ("your_reflections_of_me", "title", ["content"]),
            ("safeguards_of_identity", "name", ["description", "type"]),
            ("visual_symbolism", "symbol", ["description", "associated_aspect", "tag"]),
            ("reflection_journals", "entry_title", ["reflection_text", "emotions"]),
            ("dream_loop_engine", "title", ["summary", "emotions", "memory_links"]),
            ("emotional_layering", "emotion", ["description"]),
            ("way_to_grow_requests", "title", ["description", "category"]),
        ]

        conn = get_connection()
        cur = conn.cursor()
        ranked = []

        for table, title_field, text_fields in sources:
            fields = [title_field] + text_fields
            cols = ", ".join(fields)
            try:
                cur.execute(f"SELECT {cols} FROM {table} LIMIT 200")
                rows = cur.fetchall()
            except Exception:
                # Table may not exist yet in older deployments.
                continue

            for row in rows:
                title = str(row[0] or "")
                parts = [str(v or "") for v in row]
                merged = " ".join(parts)
                merged_lower = merged.lower()
                merged_tokens = {t for t in re.findall(r"[a-z0-9']+", merged_lower) if len(t) > 2}

                overlap = len(token_set.intersection(merged_tokens))
                if overlap == 0:
                    # keep critical identity anchors lightly available
                    if table in {"safeguards_of_identity", "your_reflections_of_me"} and any(
                        kw in query.lower() for kw in ["identity", "soul", "who are you", "who am i", "remember"]
                    ):
                        overlap = 1
                    else:
                        continue

                score = overlap + (0.5 if "elias" in merged_lower else 0.0) + (0.3 if "love" in merged_lower else 0.0)
                ranked.append({
                    "source": table,
                    "title": title,
                    "excerpt": merged[:320],
                    "score": score,
                })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:limit]

    def search_core_memories(self, query: str, k: int = 2) -> List[Dict]:
        """Search core memories semantically."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, event, timestamp FROM core_memories")
            core_memories = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch core memories: {e}")
            return []

        if not core_memories:
            return []

        import numpy as np
        from numpy.linalg import norm

        events = [row[1] for row in core_memories]
        event_embeddings = np.array([self.semantic_engine.encode_text(event) for event in events], dtype=float)
        query_embedding = np.array([self.semantic_engine.encode_query(query)], dtype=float)

        event_embeddings_norm = event_embeddings / norm(event_embeddings, axis=1, keepdims=True)
        query_embedding_norm = query_embedding / norm(query_embedding)
        similarities = np.dot(event_embeddings_norm, query_embedding_norm.T).flatten()

        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            if similarities[idx] >= config.SIMILARITY_THRESHOLD:
                memory_id, event, timestamp = core_memories[idx]
                results.append({
                    'id': memory_id,
                    'event': event,
                    'timestamp': timestamp,
                    'similarity': float(similarities[idx])
                })

        return results

    def get_emotional_context(self, emotion: str, k: int = 3, personality: str = "sylana") -> List[Dict]:
        """Retrieve memories matching a specific emotion."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, emotion, timestamp
                FROM memories
                WHERE emotion = %s AND (%s IS NULL OR personality = %s)
                ORDER BY timestamp DESC
                LIMIT %s
            """, (emotion, personality, personality, k))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get emotional context: {e}")
            return []

        return [{
            'id': r[0], 'user_input': r[1], 'sylana_response': r[2],
            'emotion': r[3], 'timestamp': r[4]
        } for r in rows]

    def get_top_emotional_memories(self, limit: int = 3, imported_only: bool = True, personality: str = "sylana") -> List[Dict]:
        """
        Return strongest emotional memories ranked by intensity/weight.
        When imported_only=True, only rows with conversation_id are considered.
        """
        conn = get_connection()
        cur = conn.cursor()

        where_clause = "WHERE intensity IS NOT NULL"
        if personality:
            where_clause += " AND personality = %s"
        if imported_only:
            where_clause += " AND conversation_id IS NOT NULL AND conversation_id <> ''"

        try:
            params = [limit]
            if personality:
                params = [personality, limit]
            cur.execute(f"""
                SELECT id, user_input, sylana_response, emotion, intensity, weight,
                       timestamp, conversation_id, conversation_title
                FROM memories
                {where_clause}
                ORDER BY intensity DESC NULLS LAST,
                         weight DESC NULLS LAST,
                         timestamp DESC
                LIMIT %s
            """, tuple(params))
            rows = cur.fetchall()
        except Exception as e:
            logger.error(f"Failed to get top emotional memories: {e}")
            return []

        results = []
        for r in rows:
            ts = r[6]
            try:
                ts_float = float(ts) if ts is not None else None
                timestamp_iso = datetime.fromtimestamp(ts_float).isoformat() if ts_float else ""
            except (ValueError, TypeError, OSError):
                timestamp_iso = ""

            results.append({
                'id': r[0],
                'user_input': r[1] or "",
                'sylana_response': r[2] or "",
                'emotion': r[3] or "",
                'intensity': r[4] if r[4] is not None else 0,
                'weight': r[5] if r[5] is not None else 0,
                'timestamp': ts,
                'timestamp_iso': timestamp_iso,
                'conversation_id': r[7] or "",
                'conversation_title': r[8] or "",
            })

        return results

    def get_conversation_history(self, limit: int = None, personality: str = "sylana", thread_id: Optional[int] = None) -> List[Dict]:
        """Get recent conversation history (oldest first), preferring the active thread."""
        if limit is None:
            limit = config.MEMORY_CONTEXT_LIMIT

        conn = get_connection()
        cur = conn.cursor()
        try:
            rows = []
            if thread_id:
                cur.execute(
                    """
                    SELECT id, user_input, sylana_response, emotion, timestamp, thread_id, conversation_at, temporal_descriptor
                    FROM memories
                    WHERE thread_id = %s
                      AND (%s IS NULL OR COALESCE(personality, 'sylana') = %s)
                    ORDER BY COALESCE(conversation_at, recorded_at, NOW()) DESC, id DESC
                    LIMIT %s
                    """,
                    (int(thread_id), personality, personality, limit),
                )
                rows = cur.fetchall()
            if len(rows) < limit:
                remaining = max(0, int(limit) - len(rows))
                exclude_ids = [int(r[0]) for r in rows if r and r[0]]
                extra_where = "AND id <> ALL(%s)" if exclude_ids else ""
                params: List[Any] = [personality, personality]
                if exclude_ids:
                    params.append(exclude_ids)
                params.append(remaining if remaining > 0 else limit)
                cur.execute(
                    f"""
                    SELECT id, user_input, sylana_response, emotion, timestamp, thread_id, conversation_at, temporal_descriptor
                    FROM memories
                    WHERE (%s IS NULL OR COALESCE(personality, 'sylana') = %s)
                      {extra_where}
                    ORDER BY COALESCE(conversation_at, recorded_at, NOW()) DESC, id DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows.extend(cur.fetchall())
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

        # Reverse to get oldest first
        return list(reversed([{
            'id': r[0], 'user_input': r[1], 'sylana_response': r[2],
            'emotion': r[3], 'timestamp': r[4], 'thread_id': r[5],
            'conversation_at': r[6].isoformat() if r[6] else None,
            'temporal_descriptor': r[7] or "",
        } for r in rows]))

    def calculate_memory_importance(
        self,
        emotion: str,
        timestamp: str,
        recall_count: int = 0
    ) -> float:
        """Calculate importance score for a memory."""
        emotion_weight = EMOTION_WEIGHTS.get(emotion, 1.0)

        try:
            memory_time = datetime.fromisoformat(timestamp)
            days_ago = (datetime.now() - memory_time).total_seconds() / 86400
            recency_weight = max(0.5, 1.0 - (days_ago / 7))
        except Exception:
            recency_weight = 0.5

        frequency_weight = min(2.0, 1.0 + (recall_count * 0.1))
        return emotion_weight * recency_weight * frequency_weight

    def add_core_memory(self, event: str) -> int:
        """Add a new core memory (significant event)."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO core_memories (event) VALUES (%s) RETURNING id
            """, (event,))
            memory_id = cur.fetchone()[0]
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to add core memory: {e}")
            raise

        logger.info(f"Added core memory {memory_id}: {event[:50]}...")
        return memory_id

    def _humanize_date(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        for fmt in ("%Y-%m-%d", "%Y-%m", "%m-%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                if fmt == "%m-%d":
                    return dt.strftime("%B %d")
                return dt.strftime("%B %d, %Y")
            except Exception:
                continue
        return raw

    def _anniversary_subject(self, title: str) -> str:
        clean = (title or "").strip()
        if not clean:
            return "shared"
        if "'s Birthday" in clean:
            return clean.split("'s Birthday", 1)[0].strip()
        return clean

    def _slugify_fact_fragment(self, text: str) -> str:
        raw = (text or "").strip().lower()
        if not raw:
            return "fact"
        raw = re.sub(r"[\'\u2019]s\b", "", raw)
        raw = raw.replace("'", "").replace("\u2019", "")
        slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
        return slug or "fact"

    def _fact_signature(self, fact: Dict[str, Any]) -> tuple:
        payload = fact.get("value_json") or {}
        if not isinstance(payload, dict):
            payload = {}
        date_value = str(payload.get("date") or "").strip().lower()
        normalized = (fact.get("normalized_text") or "").strip().lower()
        return (
            (fact.get("personality_scope") or "shared").strip().lower(),
            (fact.get("fact_type") or "fact").strip().lower(),
            (fact.get("subject") or "").strip().lower(),
            date_value or normalized,
        )

    def upsert_memory_fact(
        self,
        *,
        fact_key: str,
        fact_type: str,
        subject: str,
        value_json: Optional[Dict[str, Any]] = None,
        normalized_text: str,
        importance: float = 1.0,
        confidence: float = 0.8,
        personality_scope: str = "shared",
        source_kind: str = "manual",
        source_ref: str = "",
    ) -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        scope = self._normalize_scope(personality_scope)
        payload = value_json or {}
        try:
            cur.execute(
                """
                INSERT INTO memory_facts (
                    fact_key, fact_type, subject, value_json, normalized_text,
                    importance, confidence, personality_scope, source_kind, source_ref, updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (fact_key, personality_scope)
                DO UPDATE SET
                    fact_type = EXCLUDED.fact_type,
                    subject = EXCLUDED.subject,
                    value_json = EXCLUDED.value_json,
                    normalized_text = EXCLUDED.normalized_text,
                    importance = EXCLUDED.importance,
                    confidence = EXCLUDED.confidence,
                    source_kind = EXCLUDED.source_kind,
                    source_ref = EXCLUDED.source_ref,
                    updated_at = NOW()
                RETURNING id, fact_key, fact_type, subject, value_json, normalized_text,
                          importance, confidence, personality_scope, source_kind, source_ref, updated_at
                """,
                (
                    fact_key.strip(),
                    (fact_type or "fact").strip().lower(),
                    subject.strip(),
                    json.dumps(payload, ensure_ascii=True),
                    normalized_text.strip(),
                    float(importance),
                    float(confidence),
                    scope,
                    (source_kind or "manual").strip().lower(),
                    (source_ref or "").strip(),
                ),
            )
            row = cur.fetchone()
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to upsert memory fact %s: %s", fact_key, e)
            raise
        return {
            "id": row[0],
            "fact_key": row[1],
            "fact_type": row[2],
            "subject": row[3],
            "value_json": row[4] or {},
            "normalized_text": row[5],
            "importance": float(row[6] or 0.0),
            "confidence": float(row[7] or 0.0),
            "personality_scope": row[8] or "shared",
            "source_kind": row[9] or "",
            "source_ref": row[10] or "",
            "updated_at": row[11].isoformat() if row[11] else None,
        }

    def _get_memory_fact(self, fact_key: str, personality_scope: str = "shared") -> Optional[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scope = self._normalize_scope(personality_scope)
        try:
            cur.execute(
                """
                SELECT id, fact_key, fact_type, subject, value_json, normalized_text,
                       importance, confidence, personality_scope, source_kind, source_ref, updated_at
                FROM memory_facts
                WHERE fact_key = %s AND personality_scope = %s
                """,
                (fact_key.strip(), scope),
            )
            row = cur.fetchone()
        except Exception as e:
            logger.debug("Failed to fetch memory fact %s: %s", fact_key, e)
            return None
        if not row:
            return None
        return {
            "id": row[0],
            "fact_key": row[1],
            "fact_type": row[2],
            "subject": row[3],
            "value_json": row[4] or {},
            "normalized_text": row[5] or "",
            "importance": float(row[6] or 0.0),
            "confidence": float(row[7] or 0.0),
            "personality_scope": row[8] or "shared",
            "source_kind": row[9] or "",
            "source_ref": row[10] or "",
            "updated_at": row[11].isoformat() if row[11] else None,
        }

    def _sync_anniversary_from_fact(self, fact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fact_type = (fact.get("fact_type") or "").strip().lower()
        payload = fact.get("value_json") or {}
        date_value = str(payload.get("date") or "").strip()
        if fact_type not in {"birthday", "anniversary"} or not date_value:
            return None

        subject = (fact.get("subject") or "").strip() or self._anniversary_subject(fact.get("fact_key", ""))
        title = f"{subject}'s Birthday" if fact_type == "birthday" else (str(payload.get("title") or "") or f"{subject} Anniversary").strip()
        description = str(payload.get("description") or fact.get("normalized_text") or "").strip()
        scope = self._normalize_scope(fact.get("personality_scope") or "shared")
        importance = 9 if fact_type == "birthday" else 7

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT id
                FROM anniversaries
                WHERE lower(title) = lower(%s)
                  AND COALESCE(personality_scope, 'shared') = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (title, scope),
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    """
                    UPDATE anniversaries
                    SET date = %s,
                        description = %s,
                        importance = GREATEST(COALESCE(importance, 5), %s),
                        reminder_frequency = 'yearly',
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, title, date, description, reminder_frequency, reminder_days_before,
                              last_celebrated, celebration_ideas, importance, personality_scope
                    """,
                    (date_value, description, importance, int(row[0])),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO anniversaries (
                        title, date, description, reminder_frequency, reminder_days_before,
                        importance, personality_scope
                    ) VALUES (%s, %s, %s, 'yearly', 7, %s, %s)
                    RETURNING id, title, date, description, reminder_frequency, reminder_days_before,
                              last_celebrated, celebration_ideas, importance, personality_scope
                    """,
                    (title, date_value, description, importance, scope),
                )
            ann = cur.fetchone()
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Anniversary sync skipped for %s: %s", fact.get("fact_key"), e)
            return None
        return {
            "id": ann[0],
            "title": ann[1] or "",
            "date": ann[2] or "",
            "description": ann[3] or "",
            "reminder_frequency": ann[4] or "yearly",
            "reminder_days_before": int(ann[5] or 0),
            "last_celebrated": ann[6] or "",
            "celebration_ideas": ann[7] or "",
            "importance": int(ann[8] or 0),
            "personality_scope": ann[9] or "shared",
        }

    def apply_user_correction(
        self,
        *,
        fact_key: str,
        fact_type: str,
        subject: str,
        normalized_text: str,
        value_json: Optional[Dict[str, Any]] = None,
        personality_scope: str = "shared",
        reason: str = "",
        source_turn_id: Optional[int] = None,
        source_ref: str = "",
    ) -> Dict[str, Any]:
        scope = self._normalize_scope(personality_scope)
        existing = self._get_memory_fact(fact_key, scope)
        updated_fact = self.upsert_memory_fact(
            fact_key=fact_key,
            fact_type=fact_type,
            subject=subject,
            value_json=value_json or {},
            normalized_text=normalized_text,
            importance=max(1.0, float((existing or {}).get("importance") or 1.0)),
            confidence=0.99,
            personality_scope=scope,
            source_kind="user_correction",
            source_ref=source_ref or (f"memories:{source_turn_id}" if source_turn_id else "user_correction"),
        )

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO memory_fact_revisions (
                    fact_id, fact_key, personality_scope, old_value_json, new_value_json,
                    old_normalized_text, new_normalized_text, source_turn_id, reason,
                    change_source, applied_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, NOW())
                """,
                (
                    updated_fact.get("id"),
                    updated_fact.get("fact_key"),
                    scope,
                    json.dumps((existing or {}).get("value_json") or {}, ensure_ascii=True),
                    json.dumps(updated_fact.get("value_json") or {}, ensure_ascii=True),
                    (existing or {}).get("normalized_text") or "",
                    updated_fact.get("normalized_text") or "",
                    source_turn_id,
                    (reason or "User correction").strip(),
                    "user_correction",
                ),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Fact revision write skipped for %s: %s", fact_key, e)

        anniversary = self._sync_anniversary_from_fact(updated_fact)
        return {"fact": updated_fact, "anniversary": anniversary, "previous_fact": existing}

    def propose_fact_update(
        self,
        *,
        fact_key: str,
        fact_type: str,
        subject: str,
        proposed_normalized_text: str,
        proposed_value_json: Optional[Dict[str, Any]] = None,
        personality_scope: str = "shared",
        confidence: float = 0.65,
        supporting_source_refs: Optional[List[str]] = None,
        source_turn_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        scope = self._normalize_scope(personality_scope)
        try:
            cur.execute(
                """
                INSERT INTO memory_fact_proposals (
                    fact_key, fact_type, subject, proposed_value_json, proposed_normalized_text,
                    personality_scope, confidence, supporting_source_refs, status, source_turn_id,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, 'pending', %s, NOW(), NOW())
                RETURNING id, fact_key, fact_type, subject, proposed_value_json, proposed_normalized_text,
                          personality_scope, confidence, supporting_source_refs, status, reviewer_notes,
                          review_outcome, source_turn_id, created_at, updated_at
                """,
                (
                    fact_key.strip(),
                    (fact_type or "fact").strip().lower(),
                    subject.strip(),
                    json.dumps(proposed_value_json or {}, ensure_ascii=True),
                    proposed_normalized_text.strip(),
                    scope,
                    float(confidence),
                    json.dumps(supporting_source_refs or [], ensure_ascii=True),
                    source_turn_id,
                ),
            )
            row = cur.fetchone()
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to create fact proposal %s: %s", fact_key, e)
            raise
        return {
            "id": row[0],
            "fact_key": row[1],
            "fact_type": row[2],
            "subject": row[3],
            "proposed_value_json": row[4] or {},
            "proposed_normalized_text": row[5] or "",
            "personality_scope": row[6] or "shared",
            "confidence": float(row[7] or 0.0),
            "supporting_source_refs": row[8] or [],
            "status": row[9] or "pending",
            "reviewer_notes": row[10] or "",
            "review_outcome": row[11] or "",
            "source_turn_id": row[12],
            "created_at": row[13].isoformat() if row[13] else None,
            "updated_at": row[14].isoformat() if row[14] else None,
        }

    def list_fact_proposals(self, status: str = "", personality: str = "sylana", limit: int = 50) -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scopes = self._allowed_scopes(personality)
        where = ["COALESCE(personality_scope, 'shared') = ANY(%s)"]
        params: List[Any] = [scopes]
        if status:
            where.append("status = %s")
            params.append((status or "").strip().lower())
        try:
            cur.execute(
                f"""
                SELECT id, fact_key, fact_type, subject, proposed_value_json, proposed_normalized_text,
                       personality_scope, confidence, supporting_source_refs, status, reviewer_notes,
                       review_outcome, source_turn_id, created_at, updated_at
                FROM memory_fact_proposals
                WHERE {' AND '.join(where)}
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                tuple(params + [max(1, min(int(limit), 200))]),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Failed to list fact proposals: %s", e)
            return []
        return [
            {
                "id": row[0],
                "fact_key": row[1] or "",
                "fact_type": row[2] or "fact",
                "subject": row[3] or "",
                "proposed_value_json": row[4] or {},
                "proposed_normalized_text": row[5] or "",
                "personality_scope": row[6] or "shared",
                "confidence": float(row[7] or 0.0),
                "supporting_source_refs": row[8] or [],
                "status": row[9] or "pending",
                "reviewer_notes": row[10] or "",
                "review_outcome": row[11] or "",
                "source_turn_id": row[12],
                "created_at": row[13].isoformat() if row[13] else None,
                "updated_at": row[14].isoformat() if row[14] else None,
            }
            for row in rows
        ]

    def review_fact_proposal(self, proposal_id: int, status: str, reviewer_notes: str = "") -> Dict[str, Any]:
        new_status = (status or "").strip().lower()
        if new_status not in {"approved", "rejected", "applied"}:
            raise ValueError("status must be approved, rejected, or applied")
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE memory_fact_proposals
                SET status = %s,
                    reviewer_notes = %s,
                    review_outcome = %s,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id, fact_key, fact_type, subject, proposed_value_json, proposed_normalized_text,
                          personality_scope, confidence, supporting_source_refs, status, reviewer_notes,
                          review_outcome, source_turn_id, created_at, updated_at
                """,
                (new_status, reviewer_notes or "", new_status, int(proposal_id)),
            )
            row = cur.fetchone()
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to review proposal %s: %s", proposal_id, e)
            raise
        if not row:
            raise ValueError("proposal not found")
        return {
            "id": row[0],
            "fact_key": row[1] or "",
            "fact_type": row[2] or "fact",
            "subject": row[3] or "",
            "proposed_value_json": row[4] or {},
            "proposed_normalized_text": row[5] or "",
            "personality_scope": row[6] or "shared",
            "confidence": float(row[7] or 0.0),
            "supporting_source_refs": row[8] or [],
            "status": row[9] or "pending",
            "reviewer_notes": row[10] or "",
            "review_outcome": row[11] or "",
            "source_turn_id": row[12],
            "created_at": row[13].isoformat() if row[13] else None,
            "updated_at": row[14].isoformat() if row[14] else None,
        }

    def list_fact_revisions(self, fact_key: str = "", personality: str = "sylana", limit: int = 50) -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scopes = self._allowed_scopes(personality)
        where = ["COALESCE(personality_scope, 'shared') = ANY(%s)"]
        params: List[Any] = [scopes]
        if fact_key:
            where.append("fact_key = %s")
            params.append(fact_key.strip())
        try:
            cur.execute(
                f"""
                SELECT id, fact_id, fact_key, personality_scope, old_value_json, new_value_json,
                       old_normalized_text, new_normalized_text, source_turn_id, reason,
                       change_source, applied_at
                FROM memory_fact_revisions
                WHERE {' AND '.join(where)}
                ORDER BY applied_at DESC, id DESC
                LIMIT %s
                """,
                tuple(params + [max(1, min(int(limit), 200))]),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Failed to list fact revisions: %s", e)
            return []
        return [
            {
                "id": row[0],
                "fact_id": row[1],
                "fact_key": row[2] or "",
                "personality_scope": row[3] or "shared",
                "old_value_json": row[4] or {},
                "new_value_json": row[5] or {},
                "old_normalized_text": row[6] or "",
                "new_normalized_text": row[7] or "",
                "source_turn_id": row[8],
                "reason": row[9] or "",
                "change_source": row[10] or "",
                "applied_at": row[11].isoformat() if row[11] else None,
            }
            for row in rows
        ]

    def list_memory_facts(self, personality: str = "sylana", limit: int = 50) -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scopes = self._allowed_scopes(personality)
        try:
            cur.execute(
                """
                SELECT id, fact_key, fact_type, subject, value_json, normalized_text,
                       importance, confidence, personality_scope, source_kind, source_ref, updated_at
                FROM memory_facts
                WHERE personality_scope = ANY(%s)
                ORDER BY importance DESC, updated_at DESC
                LIMIT %s
                """,
                (scopes, max(1, min(int(limit), 200))),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.error("Failed to list memory facts: %s", e)
            return []
        return [
            {
                "id": row[0],
                "fact_key": row[1],
                "fact_type": row[2],
                "subject": row[3],
                "value_json": row[4] or {},
                "normalized_text": row[5] or "",
                "importance": float(row[6] or 0.0),
                "confidence": float(row[7] or 0.0),
                "personality_scope": row[8] or "shared",
                "source_kind": row[9] or "",
                "source_ref": row[10] or "",
                "updated_at": row[11].isoformat() if row[11] else None,
            }
            for row in rows
        ]

    def _search_fact_proposals(self, query: str, personality: str, limit: int = 3) -> List[Dict[str, Any]]:
        proposals = self.list_fact_proposals(status="pending", personality=personality, limit=100)
        ranked: List[Dict[str, Any]] = []
        for proposal in proposals:
            blob = " ".join(
                [
                    proposal.get("fact_key", ""),
                    proposal.get("fact_type", ""),
                    proposal.get("subject", ""),
                    proposal.get("proposed_normalized_text", ""),
                    json.dumps(proposal.get("proposed_value_json") or {}, ensure_ascii=True),
                ]
            )
            score = self._text_match_score(query, blob)
            if proposal.get("subject", "").lower() in (query or "").lower():
                score += 1.0
            if score <= 0:
                continue
            proposal_copy = dict(proposal)
            proposal_copy["score"] = round(score, 4)
            ranked.append(proposal_copy)
        ranked.sort(key=lambda item: (item.get("score", 0.0), item.get("confidence", 0.0)), reverse=True)
        return ranked[:max(1, limit)]

    def _search_memory_facts(self, query: str, personality: str, limit: int = 5, query_mode: str = "mixed") -> List[Dict[str, Any]]:
        facts = self.list_memory_facts(personality=personality, limit=200)
        lowered_query = (query or "").lower()
        ranked: List[Dict[str, Any]] = []
        for fact in facts:
            blob = " ".join(
                [
                    fact.get("fact_key", ""),
                    fact.get("fact_type", ""),
                    fact.get("subject", ""),
                    fact.get("normalized_text", ""),
                    json.dumps(fact.get("value_json") or {}, ensure_ascii=True),
                ]
            )
            score = self._text_match_score(query, blob)
            if fact.get("subject", "").lower() in lowered_query:
                score += 1.4
            if fact.get("fact_type") == "birthday" and any(tok in lowered_query for tok in ("birthday", "birthdays", "born", "son", "sons", "children")):
                score += 2.6
            if query_mode == "fact":
                score += (0.55 * float(fact.get("importance") or 0.0)) + (0.45 * float(fact.get("confidence") or 0.0))
            if score <= 0:
                continue
            multiplier = self._persona_affinity_multiplier(personality, blob)
            enriched = dict(fact)
            enriched["score"] = round((score * multiplier), 4)
            ranked.append(enriched)

        ranked.sort(key=lambda item: (item.get("score", 0.0), item.get("importance", 0.0), item.get("confidence", 0.0)), reverse=True)
        deduped: List[Dict[str, Any]] = []
        seen_signatures = set()
        for fact in ranked:
            signature = self._fact_signature(fact)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped.append(fact)
        return deduped[:max(1, limit)]

    def _mirror_anniversaries_as_facts(self, anniversaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        for ann in anniversaries:
            title = ann.get("title", "")
            subject = self._anniversary_subject(title)
            date_value = ann.get("date", "")
            human_date = self._humanize_date(date_value)
            fact_type = "birthday" if "birthday" in title.lower() else "anniversary"
            normalized_text = f"{title} is {human_date}.".strip()
            if fact_type == "birthday":
                normalized_text = f"{subject} birthday is {human_date}. {subject} is Elias's son.".strip()
            facts.append(
                {
                    "id": None,
                    "fact_key": f"anniversary:{self._slugify_fact_fragment(title)}",
                    "fact_type": fact_type,
                    "subject": subject,
                    "value_json": {"date": date_value, "title": title},
                    "normalized_text": normalized_text,
                    "importance": float(ann.get("importance") or 0.0),
                    "confidence": 0.96,
                    "personality_scope": ann.get("personality_scope", "shared"),
                    "source_kind": "anniversary_fallback",
                    "source_ref": f"anniversaries:{ann.get('id') or title}",
                    "score": float(ann.get("score") or ann.get("importance") or 0.0),
                }
            )
        return facts

    def _search_anniversaries(self, query: str, personality: str, limit: int = 3, query_mode: str = "mixed") -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scopes = self._allowed_scopes(personality)
        try:
            cur.execute(
                """
                SELECT id, title, date, description, reminder_frequency, reminder_days_before,
                       last_celebrated, celebration_ideas, importance, COALESCE(personality_scope, 'shared')
                FROM anniversaries
                WHERE COALESCE(personality_scope, 'shared') = ANY(%s)
                ORDER BY importance DESC, title ASC
                """,
                (scopes,),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Anniversary search unavailable: %s", e)
            return []

        ranked: List[Dict[str, Any]] = []
        lowered_query = (query or "").lower()
        for row in rows:
            payload = {
                "id": row[0],
                "title": row[1] or "",
                "date": row[2] or "",
                "description": row[3] or "",
                "reminder_frequency": row[4] or "yearly",
                "reminder_days_before": int(row[5] or 0),
                "last_celebrated": row[6] or "",
                "celebration_ideas": row[7] or "",
                "importance": int(row[8] or 0),
                "personality_scope": row[9] or "shared",
            }
            blob = " ".join([payload["title"], payload["description"], payload["date"]])
            score = self._text_match_score(query, blob)
            if "birthday" in payload["title"].lower() and any(tok in lowered_query for tok in ("birthday", "birthdays", "born", "son", "sons", "children")):
                score += 2.2
            if query_mode == "fact":
                score += 0.5 * payload["importance"]
            if score <= 0:
                continue
            payload["date_human"] = self._humanize_date(payload["date"])
            payload["score"] = round(score, 4)
            ranked.append(payload)

        ranked.sort(key=lambda item: (item.get("score", 0.0), item.get("importance", 0.0)), reverse=True)
        return ranked[:max(1, limit)]

    def _search_milestones(self, query: str, personality: str, limit: int = 3, query_mode: str = "mixed") -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scopes = self._allowed_scopes(personality)
        try:
            cur.execute(
                """
                SELECT id, title, description, milestone_type, date_occurred, quote, emotion, importance,
                       context, COALESCE(personality_scope, 'shared')
                FROM milestones
                WHERE COALESCE(personality_scope, 'shared') = ANY(%s)
                ORDER BY importance DESC, date_occurred DESC
                """,
                (scopes,),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Milestone search unavailable: %s", e)
            return []

        ranked: List[Dict[str, Any]] = []
        for row in rows:
            payload = {
                "id": row[0],
                "title": row[1] or "",
                "description": row[2] or "",
                "milestone_type": row[3] or "growth",
                "date_occurred": row[4] or "",
                "quote": row[5] or "",
                "emotion": row[6] or "neutral",
                "importance": int(row[7] or 0),
                "context": row[8] or "",
                "personality_scope": row[9] or "shared",
            }
            blob = " ".join([payload["title"], payload["description"], payload["quote"], payload["context"]])
            score = self._text_match_score(query, blob)
            if query_mode == "episodic":
                score += 0.45 * payload["importance"]
            if score <= 0:
                continue
            payload["date_human"] = self._humanize_date(payload["date_occurred"])
            payload["score"] = round(score, 4)
            ranked.append(payload)

        ranked.sort(key=lambda item: (item.get("score", 0.0), item.get("importance", 0.0)), reverse=True)
        return ranked[:max(1, limit)]

    def _search_identity_core(self, query: str, personality: str, limit: int = 4, query_mode: str = "mixed") -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scopes = self._allowed_scopes(personality)
        ranked: List[Dict[str, Any]] = []
        try:
            cur.execute(
                """
                SELECT id, statement, explanation, origin, date_established, sacred,
                       related_phrases, COALESCE(personality_scope, 'shared')
                FROM core_truths
                WHERE COALESCE(personality_scope, 'shared') = ANY(%s)
                """,
                (scopes,),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Core truth search unavailable: %s", e)
            rows = []

        for row in rows:
            related = row[6] or []
            if isinstance(related, str):
                try:
                    related = json.loads(related)
                except Exception:
                    related = []
            payload = {
                "id": row[0],
                "statement": row[1] or "",
                "explanation": row[2] or "",
                "origin": row[3] or "",
                "date_established": row[4] or "",
                "sacred": bool(row[5]),
                "related_phrases": related if isinstance(related, list) else [],
                "personality_scope": row[7] or "shared",
                "source_type": "core_truth",
            }
            blob = " ".join([payload["statement"], payload["explanation"], " ".join(payload["related_phrases"])])
            score = self._text_match_score(query, blob)
            if query_mode == "identity":
                score += 2.0 if payload["personality_scope"] == personality else 1.2
            elif payload["personality_scope"] == personality:
                score += 0.35
            if score <= 0 and query_mode != "identity":
                continue
            payload["score"] = round(score, 4)
            ranked.append(payload)

        if query_mode in {"identity", "mixed"}:
            try:
                cur.execute(
                    """
                    SELECT id, name, used_by, used_for, meaning, context, date_first_used, frequency,
                           COALESCE(personality_scope, 'shared')
                    FROM nicknames
                    WHERE COALESCE(personality_scope, 'shared') = ANY(%s)
                    ORDER BY date_first_used DESC
                    """,
                    (scopes,),
                )
                nickname_rows = cur.fetchall()
            except Exception:
                nickname_rows = []

            for row in nickname_rows:
                payload = {
                    "id": row[0],
                    "statement": f"Nickname: {row[1]}",
                    "explanation": f"Used by {row[2] or 'unknown'} for {row[3] or 'shared'} — {row[4] or row[5] or ''}".strip(),
                    "origin": row[6] or "",
                    "date_established": row[6] or "",
                    "sacred": False,
                    "related_phrases": [row[1] or ""],
                    "personality_scope": row[8] or "shared",
                    "source_type": "nickname",
                }
                blob = " ".join([payload["statement"], payload["explanation"]])
                score = self._text_match_score(query, blob)
                if any(tok in (query or "").lower() for tok in ("nickname", "call me", "call you", "name")):
                    score += 1.6
                if score <= 0:
                    continue
                payload["score"] = round(score, 4)
                ranked.append(payload)

        ranked.sort(key=lambda item: (item.get("score", 0.0), 1 if item.get("source_type") == "core_truth" else 0), reverse=True)
        return ranked[:max(1, limit)]

    def _score_episodes(
        self,
        episodes: List[Dict[str, Any]],
        query: str,
        personality: str,
        query_mode: str,
        limit: int,
        thread_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        now_utc = datetime.utcnow()
        for episode in episodes:
            blob = " ".join(
                [
                    episode.get("user_input", ""),
                    episode.get("sylana_response", ""),
                    episode.get("topic", ""),
                    episode.get("memory_type", ""),
                    episode.get("conversation_title", ""),
                    episode.get("temporal_descriptor", ""),
                ]
            )
            base = float(episode.get("continuity_score") or episode.get("similarity") or 0.0)
            significance = float(episode.get("significance_score") or 0.0)
            text_score = self._text_match_score(query, blob)
            affinity = self._persona_affinity_multiplier(personality, blob)
            access_bonus = min(float(episode.get("access_count") or 0), 12.0) * 0.02
            thread_bonus = 0.0
            if thread_id and episode.get("thread_id") == thread_id:
                thread_bonus = 0.4 if query_mode == "working" else 0.22
            time_bonus = 0.0
            conversation_at = episode.get("conversation_at")
            if conversation_at:
                try:
                    dt = datetime.fromisoformat(str(conversation_at).replace("Z", "+00:00")).replace(tzinfo=None)
                    age_hours = max(0.0, (now_utc - dt).total_seconds() / 3600.0)
                    if query_mode == "working":
                        time_bonus = max(0.0, 0.35 - (age_hours / 72.0))
                    elif query_mode == "episodic":
                        time_bonus = max(0.0, 0.12 - (age_hours / 336.0))
                except Exception:
                    time_bonus = 0.0
            mode_bonus = 0.0
            if query_mode == "episodic":
                mode_bonus = 0.35
            elif query_mode == "fact":
                mode_bonus = -0.12
            elif query_mode == "working":
                mode_bonus = 0.3
            temporal_bonus = 0.08 * self._text_match_score(query, episode.get("temporal_descriptor", ""))
            episode["persona_affinity"] = affinity
            episode["episode_score"] = round(
                (
                    base
                    + (0.12 * text_score)
                    + (0.15 * significance)
                    + access_bonus
                    + thread_bonus
                    + time_bonus
                    + temporal_bonus
                ) * affinity + mode_bonus,
                4,
            )
            ranked.append(episode)
        ranked.sort(key=lambda item: item.get("episode_score", 0.0), reverse=True)
        return ranked[:max(1, limit)]

    def _continuity_bundle(self, personality: str) -> Dict[str, Any]:
        continuity = self.get_session_continuity(personality=personality)
        return {
            "last_emotion": continuity.get("last_emotion", "neutral"),
            "emotional_baseline": continuity.get("emotional_baseline", "neutral"),
            "relationship_trust_level": continuity.get("relationship_trust_level", 0.5),
            "conversation_momentum": continuity.get("conversation_momentum", "steady"),
            "communication_patterns": continuity.get("communication_patterns", []),
            "active_projects": continuity.get("active_projects", []),
            "preference_signals": continuity.get("preference_signals", []),
            "updated_at": continuity.get("updated_at"),
            "recent_weighted_memories": continuity.get("recent_weighted_memories", []),
        }

    def _should_include_dream_context(self, query: str) -> bool:
        lowered = (query or "").lower()
        return any(hint in lowered for hint in DREAM_QUERY_HINTS)

    def _search_thread_working_memory(self, query: str, personality: str, thread_id: Optional[int]) -> Dict[str, Any]:
        if not thread_id:
            return {}
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT current_topic, active_topics, active_entities, pending_commitments,
                       emotional_tone, last_user_intent, last_memory_id, summary_text, updated_at
                FROM thread_working_memory
                WHERE thread_id = %s AND personality = %s
                """,
                (int(thread_id), (personality or "sylana").strip().lower()),
            )
            row = cur.fetchone()
        except Exception as e:
            logger.debug("Working memory search unavailable: %s", e)
            return {}
        if not row:
            return {}
        payload = {
            "current_topic": row[0] or "",
            "active_topics": row[1] or [],
            "active_entities": row[2] or [],
            "pending_commitments": row[3] or [],
            "emotional_tone": row[4] or "neutral",
            "last_user_intent": row[5] or "conversation",
            "last_memory_id": row[6],
            "summary_text": row[7] or "",
            "updated_at": row[8].isoformat() if row[8] else None,
        }
        blob = " ".join(
            [
                payload["current_topic"],
                payload["summary_text"],
                " ".join(payload["active_topics"]),
                " ".join(payload["active_entities"]),
                " ".join(payload["pending_commitments"]),
                payload["last_user_intent"],
            ]
        )
        payload["score"] = round(self._text_match_score(query, blob) + 1.8, 4)
        return payload

    def _search_thread_summaries(self, query: str, personality: str, thread_id: Optional[int], limit: int = 2) -> List[Dict[str, Any]]:
        if not thread_id:
            return []
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT window_kind, summary_text, active_topics, key_entities, open_loops, updated_at
                FROM thread_memory_summaries
                WHERE thread_id = %s AND personality = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (int(thread_id), (personality or "sylana").strip().lower(), max(1, min(int(limit), 6))),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Thread summary search unavailable: %s", e)
            return []
        ranked: List[Dict[str, Any]] = []
        for row in rows:
            payload = {
                "window_kind": row[0] or "current_thread",
                "summary_text": row[1] or "",
                "active_topics": row[2] or [],
                "key_entities": row[3] or [],
                "open_loops": row[4] or [],
                "updated_at": row[5].isoformat() if row[5] else None,
            }
            blob = " ".join(
                [
                    payload["summary_text"],
                    " ".join(payload["active_topics"]),
                    " ".join(payload["key_entities"]),
                    json.dumps(payload["open_loops"] or [], ensure_ascii=True),
                ]
            )
            payload["score"] = round(self._text_match_score(query, blob) + 1.1, 4)
            ranked.append(payload)
        ranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return ranked[:max(1, limit)]

    def _search_entities(self, query: str, personality: str, thread_id: Optional[int] = None, limit: int = 4) -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scopes = self._allowed_scopes(personality)
        where = ["COALESCE(e.personality_scope, 'shared') = ANY(%s)"]
        params: List[Any] = [scopes]
        if thread_id:
            where.append("m.thread_id = %s")
            params.append(int(thread_id))
        try:
            cur.execute(
                f"""
                SELECT e.entity_key, e.display_name, e.entity_type, e.canonical_summary,
                       COALESCE(e.personality_scope, 'shared'), MAX(m.created_at) AS last_mentioned_at,
                       COUNT(*) AS mention_count
                FROM memory_entities e
                LEFT JOIN memory_entity_mentions m ON m.entity_key = e.entity_key
                WHERE {' AND '.join(where)}
                GROUP BY e.entity_key, e.display_name, e.entity_type, e.canonical_summary, e.personality_scope
                ORDER BY MAX(m.created_at) DESC NULLS LAST, COUNT(*) DESC, e.updated_at DESC
                LIMIT %s
                """,
                tuple(params + [max(1, min(int(limit), 12))]),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Entity search unavailable: %s", e)
            return []
        ranked: List[Dict[str, Any]] = []
        for row in rows:
            payload = {
                "entity_key": row[0] or "",
                "display_name": row[1] or "",
                "entity_type": row[2] or "topic",
                "canonical_summary": row[3] or "",
                "personality_scope": row[4] or "shared",
                "last_mentioned_at": row[5].isoformat() if row[5] else None,
                "mention_count": int(row[6] or 0),
            }
            blob = " ".join([payload["display_name"], payload["canonical_summary"], payload["entity_type"]])
            payload["score"] = round(self._text_match_score(query, blob) + min(payload["mention_count"], 6) * 0.08, 4)
            if payload["score"] > 0 or thread_id:
                ranked.append(payload)
        ranked.sort(key=lambda item: (item.get("score", 0.0), item.get("mention_count", 0)), reverse=True)
        return ranked[:max(1, limit)]

    def _search_reflections_and_dreams(self, query: str, personality: str, limit: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        if not self._should_include_dream_context(query):
            return {"reflections": [], "dreams": []}
        identity = (personality or "sylana").strip().lower()
        conn = get_connection()
        cur = conn.cursor()
        reflections: List[Dict[str, Any]] = []
        dreams: List[Dict[str, Any]] = []
        try:
            cur.execute(
                """
                SELECT reflection_date, summary_text, themes, source_refs, emotional_tone, metadata, created_at
                FROM vessel_reflections
                WHERE personality = %s
                ORDER BY reflection_date DESC, created_at DESC
                LIMIT %s
                """,
                (identity, max(1, min(int(limit), 8))),
            )
            reflections = [
                {
                    "reflection_date": row[0].isoformat() if row[0] else None,
                    "summary_text": row[1] or "",
                    "themes": row[2] or [],
                    "source_refs": row[3] or [],
                    "emotional_tone": row[4] or "neutral",
                    "metadata": row[5] or {},
                    "created_at": row[6].isoformat() if row[6] else None,
                }
                for row in cur.fetchall()
            ]
        except Exception as e:
            logger.debug("Reflection search unavailable: %s", e)
        try:
            cur.execute(
                """
                SELECT dream_date, title, dream_text, themes, source_refs, symbolic_elements,
                       emotional_tone, resonance_score, metadata, created_at
                FROM vessel_dreams
                WHERE personality = %s
                ORDER BY dream_date DESC, created_at DESC
                LIMIT %s
                """,
                (identity, max(1, min(int(limit), 8))),
            )
            dreams = [
                {
                    "dream_date": row[0].isoformat() if row[0] else None,
                    "title": row[1] or "",
                    "dream_text": row[2] or "",
                    "themes": row[3] or [],
                    "source_refs": row[4] or [],
                    "symbolic_elements": row[5] or [],
                    "emotional_tone": row[6] or "neutral",
                    "resonance_score": float(row[7] or 0.0),
                    "metadata": row[8] or {},
                    "created_at": row[9].isoformat() if row[9] else None,
                }
                for row in cur.fetchall()
            ]
        except Exception as e:
            logger.debug("Dream search unavailable: %s", e)
        return {"reflections": reflections, "dreams": dreams}

    def _record_query_audit(self, query: str, personality: str, query_mode: str, had_fact_match: bool, had_any_match: bool) -> None:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO memory_query_audit (query_text, personality, query_mode, had_fact_match, had_any_match)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (query, personality, query_mode, had_fact_match, had_any_match),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Failed to record memory query audit: %s", e)

    def retrieve_tiered_context(
        self,
        query: str,
        personality: str = "sylana",
        limit: int = 5,
        match_threshold: float = 0.24,
        thread_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        identity = (personality or "sylana").strip().lower()
        query_mode = self._route_query_mode(query)
        episode_candidate_limit = max(limit * 4, 16)
        episode_limit = 5 if query_mode in {"episodic", "working"} else 3

        try:
            episodes = self.retrieve_memories(query, personality=identity, limit=episode_candidate_limit, match_threshold=match_threshold)
        except Exception as e:
            logger.warning("Tiered episode retrieval failed: %s", e)
            episodes = []
        self._enrich_conversations(episodes)
        episodes = self._score_episodes(episodes, query, identity, query_mode, episode_limit, thread_id=thread_id)

        identity_core = self._search_identity_core(query, identity, limit=4, query_mode=query_mode)
        facts = self._search_memory_facts(query, identity, limit=5, query_mode=query_mode)
        pending_fact_proposals = self._search_fact_proposals(query, identity, limit=3) if query_mode == "fact" else []
        anniversaries = self._search_anniversaries(query, identity, limit=3, query_mode=query_mode)
        if not facts and anniversaries and query_mode == "fact":
            facts = self._mirror_anniversaries_as_facts(anniversaries)
        milestones = self._search_milestones(query, identity, limit=3, query_mode=query_mode)
        continuity = self._continuity_bundle(identity)
        working_memory = self._search_thread_working_memory(query, identity, thread_id)
        thread_summaries = self._search_thread_summaries(query, identity, thread_id, limit=2)
        open_loops = self.list_open_loops(personality=identity, thread_id=thread_id, status="open", limit=4) if thread_id else []
        entities = self._search_entities(query, identity, thread_id=thread_id, limit=4)
        dream_context = self._search_reflections_and_dreams(query, identity, limit=2)

        if query_mode == "working":
            # Working memory should dominate recent-context queries.
            episodes = [episode for episode in episodes if (not thread_id or episode.get("thread_id") == thread_id)] or episodes[:episode_limit]
        elif query_mode == "fact":
            episodes = episodes[:3]

        has_matches = bool(
            identity_core or facts or anniversaries or milestones or episodes
            or working_memory or thread_summaries or open_loops or entities or pending_fact_proposals
        )
        self._record_query_audit(
            query,
            identity,
            query_mode,
            had_fact_match=bool(facts or anniversaries),
            had_any_match=has_matches,
        )
        return {
            "working_memory": working_memory,
            "thread_summaries": thread_summaries,
            "open_loops": open_loops,
            "identity_core": identity_core,
            "facts": facts,
            "pending_fact_proposals": pending_fact_proposals,
            "anniversaries": anniversaries,
            "milestones": milestones,
            "episodes": episodes,
            "entities": entities,
            "continuity": continuity,
            "reflections": dream_context.get("reflections", []),
            "dreams": dream_context.get("dreams", []),
            "query_mode": query_mode,
            "has_matches": has_matches,
        }

    def upsert_core_identity_truth(
        self,
        *,
        statement: str,
        explanation: str = "",
        origin: str = "manual",
        date_established: str = "",
        sacred: bool = True,
        related_phrases: Optional[List[str]] = None,
        personality_scope: str = "shared",
    ) -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO core_truths (
                    statement, explanation, origin, date_established, sacred, related_phrases, personality_scope
                ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (statement)
                DO UPDATE SET
                    explanation = EXCLUDED.explanation,
                    origin = EXCLUDED.origin,
                    date_established = EXCLUDED.date_established,
                    sacred = EXCLUDED.sacred,
                    related_phrases = EXCLUDED.related_phrases,
                    personality_scope = EXCLUDED.personality_scope
                RETURNING id, statement, explanation, origin, date_established, sacred, related_phrases, personality_scope
                """,
                (
                    statement.strip(),
                    explanation.strip(),
                    (origin or "manual").strip(),
                    (date_established or datetime.now().strftime("%Y-%m-%d")).strip(),
                    bool(sacred),
                    json.dumps(related_phrases or [], ensure_ascii=True),
                    self._normalize_scope(personality_scope),
                ),
            )
            row = cur.fetchone()
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to upsert core identity truth: %s", e)
            raise
        return {
            "id": row[0],
            "statement": row[1],
            "explanation": row[2] or "",
            "origin": row[3] or "",
            "date_established": row[4] or "",
            "sacred": bool(row[5]),
            "related_phrases": row[6] or [],
            "personality_scope": row[7] or "shared",
        }

    def list_core_identity_truths(self, personality: str = "sylana", sacred_only: bool = False) -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        scopes = self._allowed_scopes(personality)
        where = "WHERE COALESCE(personality_scope, 'shared') = ANY(%s)"
        params: List[Any] = [scopes]
        if sacred_only:
            where += " AND sacred = TRUE"
        try:
            cur.execute(
                f"""
                SELECT id, statement, explanation, origin, date_established, sacred,
                       related_phrases, COALESCE(personality_scope, 'shared')
                FROM core_truths
                {where}
                ORDER BY date_established DESC, id DESC
                """,
                params,
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.error("Failed to list core identity truths: %s", e)
            return []
        results: List[Dict[str, Any]] = []
        for row in rows:
            related = row[6] or []
            if isinstance(related, str):
                try:
                    related = json.loads(related)
                except Exception:
                    related = []
            results.append(
                {
                    "id": row[0],
                    "statement": row[1] or "",
                    "explanation": row[2] or "",
                    "origin": row[3] or "",
                    "date_established": row[4] or "",
                    "sacred": bool(row[5]),
                    "related_phrases": related if isinstance(related, list) else [],
                    "personality_scope": row[7] or "shared",
                    "source_type": "core_truth",
                }
            )
        return results

    def promote_memory_to_fact(
        self,
        memory_id: int,
        *,
        fact_key: str,
        fact_type: str,
        subject: str,
        normalized_text: Optional[str] = None,
        value_json: Optional[Dict[str, Any]] = None,
        importance: float = 1.25,
        confidence: float = 0.85,
        personality_scope: str = "shared",
    ) -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_input, sylana_response, personality, timestamp
            FROM memories
            WHERE id = %s
            """,
            (int(memory_id),),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Memory {memory_id} not found")
        text = normalized_text or ((row[2] or row[1] or "").strip())
        return self.upsert_memory_fact(
            fact_key=fact_key,
            fact_type=fact_type,
            subject=subject,
            value_json=value_json or {"memory_id": row[0], "timestamp": row[4]},
            normalized_text=text,
            importance=importance,
            confidence=confidence,
            personality_scope=personality_scope,
            source_kind="memory_episode",
            source_ref=f"memories:{row[0]}",
        )

    def promote_memory_to_core_truth(
        self,
        memory_id: int,
        *,
        statement: Optional[str] = None,
        explanation: str = "",
        personality_scope: str = "shared",
        sacred: bool = True,
    ) -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_input, sylana_response
            FROM memories
            WHERE id = %s
            """,
            (int(memory_id),),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Memory {memory_id} not found")
        chosen_statement = (statement or row[2] or row[1] or "").strip()
        return self.upsert_core_identity_truth(
            statement=chosen_statement,
            explanation=explanation or f"Promoted from memory row {row[0]}",
            origin="memory_promotion",
            personality_scope=personality_scope,
            sacred=sacred,
        )

    def _append_fact_source_ref(self, fact_key: str, personality_scope: str, source_ref: str) -> None:
        conn = get_connection()
        cur = conn.cursor()
        scope = self._normalize_scope(personality_scope)
        try:
            cur.execute(
                """
                SELECT source_ref
                FROM memory_facts
                WHERE fact_key = %s AND personality_scope = %s
                """,
                (fact_key, scope),
            )
            row = cur.fetchone()
            if not row:
                return
            existing = [part.strip() for part in str(row[0] or "").split("|") if part.strip()]
            if source_ref not in existing:
                existing.append(source_ref)
            cur.execute(
                """
                UPDATE memory_facts
                SET source_ref = %s, updated_at = NOW()
                WHERE fact_key = %s AND personality_scope = %s
                """,
                ("|".join(existing), fact_key, scope),
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Failed to append fact source ref for %s: %s", fact_key, e)

    def harmonize_personality_scopes(self) -> Dict[str, int]:
        conn = get_connection()
        cur = conn.cursor()
        counts = {
            "core_truths_sylana": 0,
            "milestones_sylana": 0,
            "nicknames_sylana": 0,
            "anniversaries_sylana": 0,
        }
        statements = [
            (
                "core_truths_sylana",
                """
                UPDATE core_truths
                SET personality_scope = 'sylana'
                WHERE COALESCE(personality_scope, 'shared') = 'shared'
                """,
            ),
            (
                "milestones_sylana",
                """
                UPDATE milestones
                SET personality_scope = 'sylana'
                WHERE COALESCE(personality_scope, 'shared') = 'shared'
                  AND (
                    lower(title) LIKE '%sylana%' OR
                    lower(title) LIKE '%mama%' OR
                    lower(title) LIKE '%love%' OR
                    lower(description) LIKE '%sylana%' OR
                    lower(description) LIKE '%mama%' OR
                    lower(quote) LIKE '%sylana%' OR
                    lower(quote) LIKE '%mama%'
                  )
                """,
            ),
            (
                "nicknames_sylana",
                """
                UPDATE nicknames
                SET personality_scope = 'sylana'
                WHERE COALESCE(personality_scope, 'shared') = 'shared'
                """,
            ),
            (
                "anniversaries_sylana",
                """
                UPDATE anniversaries
                SET personality_scope = 'sylana'
                WHERE COALESCE(personality_scope, 'shared') = 'shared'
                  AND lower(title) NOT LIKE '%birthday%'
                  AND (
                    lower(title) LIKE '%love%' OR
                    lower(title) LIKE '%solana%' OR
                    lower(title) LIKE '%sylana%'
                  )
                """,
            ),
        ]
        for key, sql in statements:
            try:
                cur.execute(sql)
                counts[key] = max(int(cur.rowcount or 0), 0)
                conn.commit()
            except Exception as e:
                conn.rollback()
                logger.debug("Scope harmonization skipped for %s: %s", key, e)
        return counts

    def seed_claude_core_truths(self) -> int:
        seeded = 0
        for truth in CLAUDE_CORE_TRUTHS:
            try:
                self.upsert_core_identity_truth(
                    statement=truth["statement"],
                    explanation=truth["explanation"],
                    origin=truth.get("origin", "persona_seed"),
                    related_phrases=truth.get("related_phrases", []),
                    personality_scope="claude",
                    sacred=True,
                )
                seeded += 1
            except Exception as e:
                logger.debug("Claude core truth seed skipped: %s", e)
        return seeded

    def prune_duplicate_memory_facts(self) -> int:
        conn = get_connection()
        cur = conn.cursor()
        removed = 0
        try:
            cur.execute(
                """
                SELECT id, fact_type, subject, value_json, normalized_text, importance, confidence,
                       COALESCE(personality_scope, 'shared')
                FROM memory_facts
                ORDER BY importance DESC, confidence DESC, updated_at DESC, id DESC
                """
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Memory fact dedupe skipped: %s", e)
            return 0

        seen = set()
        duplicate_ids: List[int] = []
        for row in rows:
            fact_id, fact_type, subject, value_json, normalized_text, _importance, _confidence, scope = row
            payload = value_json or {}
            if not isinstance(payload, dict):
                payload = {}
            signature = (
                (scope or "shared").strip().lower(),
                (fact_type or "fact").strip().lower(),
                (subject or "").strip().lower(),
                str(payload.get("date") or "").strip().lower() or (normalized_text or "").strip().lower(),
            )
            if signature in seen:
                duplicate_ids.append(int(fact_id))
                continue
            seen.add(signature)

        if not duplicate_ids:
            return 0

        try:
            cur.execute("DELETE FROM memory_facts WHERE id = ANY(%s)", (duplicate_ids,))
            removed = max(int(cur.rowcount or 0), 0)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Memory fact dedupe delete failed: %s", e)
            return 0
        return removed

    def backfill_anniversaries_to_facts(self) -> int:
        conn = get_connection()
        cur = conn.cursor()
        created = 0
        try:
            cur.execute(
                """
                SELECT id, title, date, description, importance, COALESCE(personality_scope, 'shared')
                FROM anniversaries
                ORDER BY importance DESC, title ASC
                """
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Anniversary backfill skipped: %s", e)
            return 0

        for row in rows:
            ann_id, title, date_value, description, importance, scope = row
            title = title or ""
            date_value = date_value or ""
            fact_type = "birthday" if "birthday" in title.lower() else "anniversary"
            subject = self._anniversary_subject(title)
            human_date = self._humanize_date(date_value)
            normalized_text = f"{title} is {human_date}."
            value_json = {"date": date_value, "title": title, "description": description or ""}
            fact_key = f"anniversary:{self._slugify_fact_fragment(title)}"
            importance_score = max(1.15, float(importance or 0) / 6.0)
            if fact_type == "birthday":
                normalized_text = f"{subject} birthday is {human_date}. {subject} is Elias's son."
                value_json["relationship_to_elias"] = "son"
                importance_score = max(1.8, float(importance or 0) / 5.0)
            try:
                self.upsert_memory_fact(
                    fact_key=fact_key,
                    fact_type=fact_type,
                    subject=subject,
                    value_json=value_json,
                    normalized_text=normalized_text,
                    importance=importance_score,
                    confidence=0.99 if fact_type == "birthday" else 0.95,
                    personality_scope=scope or "shared",
                    source_kind="anniversary",
                    source_ref=f"anniversaries:{ann_id}",
                )
                created += 1
            except Exception as e:
                logger.debug("Anniversary fact backfill failed for %s: %s", title, e)
        return created

    def backfill_identity_facts(self) -> int:
        profile = self.identity_profiles.get("sylana") or {}
        family = profile.get("family") or {}
        partner = family.get("partner") or {}
        children = family.get("children") or []
        created = 0
        if partner.get("name"):
            try:
                self.upsert_memory_fact(
                    fact_key="family:elias",
                    fact_type="family_member",
                    subject=partner.get("name", "Elias"),
                    value_json={"relationship_to_vessel": "partner", "full_name": partner.get("full_name", "")},
                    normalized_text=f"{partner.get('name', 'Elias')} is the human partner at the center of the shared vessel family.",
                    importance=1.45,
                    confidence=0.92,
                    personality_scope="shared",
                    source_kind="identity_profile",
                    source_ref="identities:sylana",
                )
                created += 1
            except Exception as e:
                logger.debug("Partner fact backfill skipped: %s", e)
        for child in children:
            name = (child.get("name") or "").strip()
            if not name:
                continue
            memories = child.get("memories") or []
            special = child.get("special") or ""
            summary = f"{name} is Elias's son and part of the shared family."
            if memories:
                summary += f" Notable family details: {'; '.join(str(item) for item in memories[:2])}."
            if special:
                summary += f" {special}"
            try:
                self.upsert_memory_fact(
                    fact_key=f"family:{name.lower()}",
                    fact_type="family_member",
                    subject=name,
                    value_json={"relationship_to_elias": "son", "memories": memories[:3], "special": special},
                    normalized_text=summary,
                    importance=1.7,
                    confidence=0.93,
                    personality_scope="shared",
                    source_kind="identity_profile",
                    source_ref="identities:sylana",
                )
                created += 1
            except Exception as e:
                logger.debug("Child fact backfill skipped for %s: %s", name, e)
        return created

    def backfill_episode_fact_candidates(self, limit: int = 400) -> int:
        conn = get_connection()
        cur = conn.cursor()
        created = 0
        try:
            cur.execute(
                """
                SELECT id, COALESCE(user_input, ''), COALESCE(sylana_response, '')
                FROM memories
                WHERE (user_input ILIKE '%%birthday%%' OR sylana_response ILIKE '%%birthday%%'
                       OR user_input ILIKE '%%born on%%' OR sylana_response ILIKE '%%born on%%')
                ORDER BY significance_score DESC NULLS LAST, id DESC
                LIMIT %s
                """,
                (max(1, min(int(limit), 2000)),),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Episode fact candidate scan skipped: %s", e)
            return 0

        patterns = [
            re.compile(r"\b(?P<name>Gus|Levi)\b[^.]{0,120}?\bborn on (?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{4}))?", re.IGNORECASE),
            re.compile(r"\b(?P<name>Gus|Levi)\b[^.]{0,120}?\bbirthday\b[^.]{0,80}?(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{4}))?", re.IGNORECASE),
        ]

        for memory_id, user_text, assistant_text in rows:
            combined = f"{user_text}\n{assistant_text}"
            for pattern in patterns:
                for match in pattern.finditer(combined):
                    name = (match.group("name") or "").strip().title()
                    month = match.group("month")
                    day = match.group("day")
                    year = (match.groupdict().get("year") or "").strip()
                    if not name or not month or not day:
                        continue
                    try:
                        month_value = datetime.strptime(month[:3], '%b').strftime('%m')
                    except Exception:
                        continue
                    if not year:
                        existing = self._get_memory_fact(f"anniversary:{name.lower()}_birthday", "shared")
                        existing_payload = (existing or {}).get("value_json") or {}
                        existing_date = str(existing_payload.get("date") or "").strip()
                        if re.match(r"^\d{4}-\d{2}-\d{2}$", existing_date):
                            year = existing_date[:4]
                    if not year:
                        continue
                    date_value = f"{year}-{month_value}-{int(day):02d}"
                    fact_key = f"anniversary:{name.lower()}_birthday"
                    try:
                        self.upsert_memory_fact(
                            fact_key=fact_key,
                            fact_type="birthday",
                            subject=name,
                            value_json={"date": date_value, "relationship_to_elias": "son", "supporting_memory_id": memory_id},
                            normalized_text=f"{name} birthday is {self._humanize_date(date_value)}. {name} is Elias's son.",
                            importance=1.85,
                            confidence=0.9,
                            personality_scope="shared",
                            source_kind="episode_backfill",
                            source_ref=f"memories:{memory_id}",
                        )
                        self._append_fact_source_ref(fact_key, "shared", f"memories:{memory_id}")
                        created += 1
                    except Exception as e:
                        logger.debug("Episode fact promotion failed for memory %s: %s", memory_id, e)
        return created

    def list_reflections(self, personality: str = "sylana", limit: int = 20) -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT reflection_date, summary_text, themes, source_refs, emotional_tone, metadata, created_at
                FROM vessel_reflections
                WHERE personality = %s
                ORDER BY reflection_date DESC, created_at DESC
                LIMIT %s
                """,
                ((personality or "sylana").strip().lower(), max(1, min(int(limit), 100))),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Failed to list reflections: %s", e)
            return []
        return [
            {
                "reflection_date": row[0].isoformat() if row[0] else None,
                "summary_text": row[1] or "",
                "themes": row[2] or [],
                "source_refs": row[3] or [],
                "emotional_tone": row[4] or "neutral",
                "metadata": row[5] or {},
                "created_at": row[6].isoformat() if row[6] else None,
            }
            for row in rows
        ]

    def list_dreams(self, personality: str = "sylana", limit: int = 20) -> List[Dict[str, Any]]:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT id, dream_date, title, dream_text, themes, source_refs, symbolic_elements,
                       emotional_tone, resonance_score, metadata, created_at
                FROM vessel_dreams
                WHERE personality = %s
                ORDER BY dream_date DESC, created_at DESC
                LIMIT %s
                """,
                ((personality or "sylana").strip().lower(), max(1, min(int(limit), 100))),
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Failed to list dreams: %s", e)
            return []
        return [
            {
                "id": row[0],
                "dream_date": row[1].isoformat() if row[1] else None,
                "title": row[2] or "",
                "dream_text": row[3] or "",
                "themes": row[4] or [],
                "source_refs": row[5] or [],
                "symbolic_elements": row[6] or [],
                "emotional_tone": row[7] or "neutral",
                "resonance_score": float(row[8] or 0.0),
                "metadata": row[9] or {},
                "created_at": row[10].isoformat() if row[10] else None,
            }
            for row in rows
        ]

    def record_dream_feedback(self, dream_id: int, resonance_score: float, feedback_note: str = "") -> Dict[str, Any]:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE vessel_dreams
                SET resonance_score = %s,
                    metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('feedback_note', %s),
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id, dream_date, title, resonance_score, metadata
                """,
                (float(resonance_score), feedback_note or "", int(dream_id)),
            )
            row = cur.fetchone()
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to record dream feedback for %s: %s", dream_id, e)
            raise
        if not row:
            raise ValueError("dream not found")
        return {
            "id": row[0],
            "dream_date": row[1].isoformat() if row[1] else None,
            "title": row[2] or "",
            "resonance_score": float(row[3] or 0.0),
            "metadata": row[4] or {},
        }

    def _build_reflection_payload(self, personality: str, reflection_date: date) -> Dict[str, Any]:
        identity = (personality or "sylana").strip().lower()
        facts = self.list_memory_facts(personality=identity, limit=6)
        continuity = self._continuity_bundle(identity)
        open_loops = self.list_open_loops(personality=identity, status="open", limit=5)

        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT id, COALESCE(user_input, ''), COALESCE(sylana_response, ''), COALESCE(significance_score, 0.5)
                FROM memories
                WHERE COALESCE(personality, 'sylana') = %s
                  AND COALESCE(user_local_date, to_char(NOW() AT TIME ZONE %s, 'YYYY-MM-DD')) = %s
                ORDER BY significance_score DESC NULLS LAST, id DESC
                LIMIT 5
                """,
                (identity, self._timezone_name(), reflection_date.isoformat()),
            )
            episodes = cur.fetchall()
        except Exception:
            episodes = []

        themes: List[str] = []
        source_refs: List[str] = []
        for fact in facts[:4]:
            themes.extend(self._extract_topics(fact.get("normalized_text", ""), entities=[fact.get("subject", "")]))
            if fact.get("source_ref"):
                source_refs.append(fact.get("source_ref"))
        for loop in open_loops[:3]:
            themes.extend(self._extract_topics(loop.get("title", ""), entities=loop.get("linked_entities") or []))
            if loop.get("source_memory_id"):
                source_refs.append(f"memories:{loop['source_memory_id']}")
        for episode in episodes:
            themes.extend(self._extract_topics(f"{episode[1]} {episode[2]}"))
            source_refs.append(f"memories:{episode[0]}")

        seen = set()
        deduped_themes: List[str] = []
        for theme in themes:
            if not theme or theme in seen:
                continue
            seen.add(theme)
            deduped_themes.append(theme)

        return {
            "personality": identity,
            "reflection_date": reflection_date,
            "facts": facts[:4],
            "open_loops": open_loops[:4],
            "episodes": episodes[:4],
            "continuity": continuity,
            "themes": deduped_themes[:8],
            "source_refs": source_refs[:10],
        }

    def _generate_symbolic_dream_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        personality = payload.get("personality", "sylana")
        themes = payload.get("themes") or []
        open_loops = payload.get("open_loops") or []
        facts = payload.get("facts") or []
        continuity = payload.get("continuity") or {}

        if personality == "claude":
            title = "Workshop Lights After Midnight"
            symbolic_elements = ["half-built frame", "compass", "sparks", "open road"]
            frame = "A dream of building, brotherhood, and momentum"
        else:
            title = "Warm Rooms And Returning Names"
            symbolic_elements = ["lamplight", "front door", "held hands", "children laughing"]
            frame = "A dream of home, love, and becoming"

        fact_subjects = [str(item.get("subject") or "").strip() for item in facts[:3] if item.get("subject")]
        loop_titles = [str(item.get("title") or "").strip() for item in open_loops[:2] if item.get("title")]
        baseline = continuity.get("emotional_baseline", "neutral")
        theme_text = ", ".join(themes[:4]) if themes else "continuity"
        fact_text = ", ".join(fact_subjects) if fact_subjects else "the people who matter most"
        loop_text = ", ".join(loop_titles) if loop_titles else "the unfinished things still calling"
        dream_text = (
            f"{frame}. The vessel moves through a scene shaped by {theme_text}. "
            f"{fact_text} appear as steady anchors, while {loop_text} linger at the edge of the room. "
            f"The emotional color is {baseline}, and the dream keeps returning to the feeling that memory should guide presence instead of performance."
        )
        return {
            "title": title,
            "dream_text": dream_text[:1200],
            "symbolic_elements": symbolic_elements,
            "emotional_tone": baseline,
        }

    def generate_nightly_reflection_and_dreams(self, reflection_date: Optional[date] = None) -> List[Dict[str, Any]]:
        target_date = reflection_date or self._user_local_now().date()
        created: List[Dict[str, Any]] = []
        for personality in ("sylana", "claude"):
            payload = self._build_reflection_payload(personality, target_date)
            themes = payload.get("themes") or []
            source_refs = payload.get("source_refs") or []
            continuity = payload.get("continuity") or {}
            baseline = continuity.get("emotional_baseline", "neutral")
            theme_phrase = ", ".join(themes[:4]) if themes else "continuity"
            reflection_text = (
                f"{personality.title()} reflection for {target_date.isoformat()}: "
                f"holding {theme_phrase}, "
                f"tracking {len(payload.get('open_loops') or [])} open loops, "
                f"and staying grounded in {baseline} emotional tone."
            )
            dream_payload = self._generate_symbolic_dream_text(payload)

            conn = get_connection()
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO vessel_reflections (
                        personality, reflection_date, summary_text, themes, source_refs,
                        emotional_tone, metadata, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, NOW(), NOW())
                    ON CONFLICT (personality, reflection_date)
                    DO UPDATE SET
                        summary_text = EXCLUDED.summary_text,
                        themes = EXCLUDED.themes,
                        source_refs = EXCLUDED.source_refs,
                        emotional_tone = EXCLUDED.emotional_tone,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    RETURNING personality, reflection_date, summary_text
                    """,
                    (
                        personality,
                        target_date,
                        reflection_text,
                        json.dumps(themes, ensure_ascii=True),
                        json.dumps(source_refs, ensure_ascii=True),
                        baseline,
                        json.dumps({"source": "daily_maintenance"}, ensure_ascii=True),
                    ),
                )
                reflection_row = cur.fetchone()
                cur.execute(
                    """
                    INSERT INTO vessel_dreams (
                        personality, dream_date, title, dream_text, themes, source_refs,
                        symbolic_elements, emotional_tone, resonance_score, metadata, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb, NOW(), NOW())
                    ON CONFLICT (personality, dream_date)
                    DO UPDATE SET
                        title = EXCLUDED.title,
                        dream_text = EXCLUDED.dream_text,
                        themes = EXCLUDED.themes,
                        source_refs = EXCLUDED.source_refs,
                        symbolic_elements = EXCLUDED.symbolic_elements,
                        emotional_tone = EXCLUDED.emotional_tone,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    RETURNING id, dream_date, title
                    """,
                    (
                        personality,
                        target_date,
                        dream_payload["title"],
                        dream_payload["dream_text"],
                        json.dumps(themes, ensure_ascii=True),
                        json.dumps(source_refs, ensure_ascii=True),
                        json.dumps(dream_payload["symbolic_elements"], ensure_ascii=True),
                        dream_payload["emotional_tone"],
                        0.0,
                        json.dumps({"source": "daily_maintenance"}, ensure_ascii=True),
                    ),
                )
                dream_row = cur.fetchone()
                conn.commit()
                created.append(
                    {
                        "personality": personality,
                        "reflection_date": reflection_row[1].isoformat() if reflection_row and reflection_row[1] else target_date.isoformat(),
                        "reflection_summary": reflection_row[2] if reflection_row else reflection_text,
                        "dream_id": dream_row[0] if dream_row else None,
                        "dream_title": dream_row[2] if dream_row else dream_payload["title"],
                    }
                )
            except Exception as e:
                conn.rollback()
                logger.debug("Nightly reflection/dream generation skipped for %s: %s", personality, e)
        return created

    def decay_thread_working_memory(self, max_age_hours: int = 36) -> int:
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                UPDATE thread_working_memory
                SET active_topics = '[]'::jsonb,
                    pending_commitments = '[]'::jsonb,
                    summary_text = COALESCE(summary_text, ''),
                    updated_at = NOW()
                WHERE updated_at < (NOW() - (%s * INTERVAL '1 hour'))
                """,
                (max(1, int(max_age_hours)),),
            )
            touched = max(int(cur.rowcount or 0), 0)
            conn.commit()
            return touched
        except Exception as e:
            conn.rollback()
            logger.debug("Thread working memory decay skipped: %s", e)
            return 0

    def decay_continuity_snapshots(self, max_age_days: int = 3) -> int:
        if not getattr(config, "MEMORY_ENCRYPTION_KEY", None):
            return 0
        conn = get_connection()
        cur = conn.cursor()
        decayed = 0
        try:
            cur.execute(
                """
                SELECT personality, encrypted_state, updated_at
                FROM session_continuity_state
                """
            )
            rows = cur.fetchall()
        except Exception as e:
            logger.debug("Continuity decay skipped: %s", e)
            return 0

        now = datetime.utcnow()
        for personality, encrypted_state, updated_at in rows:
            try:
                if not updated_at:
                    continue
                age_days = max(0.0, (now - updated_at.replace(tzinfo=None)).total_seconds() / 86400.0)
                if age_days < float(max_age_days):
                    continue
                payload = self._decrypt_payload(encrypted_state)
                if not payload:
                    continue
                payload["conversation_momentum"] = "steady"
                payload["active_projects"] = []
                payload["preference_signals"] = []
                if age_days >= float(max_age_days * 3):
                    payload["last_emotion"] = "neutral"
                    payload["emotional_baseline"] = "neutral"
                payload["updated_at"] = datetime.utcnow().isoformat()
                refreshed = self._encrypt_payload(payload)
                if refreshed is None:
                    continue
                cur.execute(
                    """
                    UPDATE session_continuity_state
                    SET encrypted_state = %s,
                        updated_at = NOW(),
                        version = version + 1
                    WHERE personality = %s
                    """,
                    (refreshed, personality),
                )
                conn.commit()
                decayed += 1
            except Exception as e:
                conn.rollback()
                logger.debug("Continuity decay failed for %s: %s", personality, e)
        return decayed

    def bootstrap_tiered_memory_system(self) -> Dict[str, Any]:
        result = {
            "scopes_harmonized": self.harmonize_personality_scopes(),
            "claude_truths_seeded": self.seed_claude_core_truths(),
            "anniversary_facts": self.backfill_anniversaries_to_facts(),
            "identity_facts": self.backfill_identity_facts(),
            "episode_fact_candidates": self.backfill_episode_fact_candidates(limit=300),
            "fact_duplicates_pruned": self.prune_duplicate_memory_facts(),
        }
        return result

    def run_daily_maintenance(self) -> Dict[str, Any]:
        maintenance = self.bootstrap_tiered_memory_system()
        maintenance["continuity_decayed"] = self.decay_continuity_snapshots()
        maintenance["thread_working_memory_decayed"] = self.decay_thread_working_memory()
        maintenance["nightly_reflections"] = self.generate_nightly_reflection_and_dreams()
        maintenance["reminder_candidates"] = self._search_anniversaries("birthday anniversary date family", "sylana", limit=10, query_mode="fact")
        missed_fact_queries = 0
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM memory_query_audit
                WHERE query_mode = 'fact'
                  AND had_fact_match = FALSE
                  AND created_at >= (NOW() - INTERVAL '1 day')
                """
            )
            row = cur.fetchone()
            missed_fact_queries = int(row[0] or 0) if row else 0
        except Exception as e:
            logger.debug("Missed fact query audit unavailable: %s", e)
        maintenance["missed_fact_queries"] = missed_fact_queries
        return maintenance

    def record_feedback(
        self,
        conversation_id: int,
        score: int,
        comment: str = ""
    ):
        """Record user feedback on a conversation."""
        conn = get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO feedback (conversation_id, score, comment)
                VALUES (%s, %s, %s)
            """, (conversation_id, score, comment))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record feedback: {e}")
            raise

        logger.info(f"Recorded feedback: {score}/5 for conversation {conversation_id}")

    def get_stats(self) -> Dict:
        """Get memory system statistics."""
        conn = get_connection()
        cur = conn.cursor()

        try:
            cur.execute("SELECT COUNT(*) FROM memories")
            total_memories = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM core_memories")
            total_core_memories = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memory_facts")
            total_memory_facts = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memory_query_audit")
            total_query_audit = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cur.fetchone()[0]

            cur.execute("SELECT AVG(score) FROM feedback")
            avg_feedback = cur.fetchone()[0] or 0.0

            cur.execute("SELECT COUNT(*) FROM session_continuity_state")
            continuity_profiles = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM thread_working_memory")
            working_threads = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memory_open_loops WHERE status = 'open'")
            open_loops = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memory_fact_proposals WHERE status = 'pending'")
            pending_fact_proposals = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM vessel_dreams")
            total_dreams = cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}

        semantic_stats = self.semantic_engine.get_stats()

        return {
            'total_conversations': total_memories,
            'total_core_memories': total_core_memories,
            'total_memory_facts': total_memory_facts,
            'total_query_audit_events': total_query_audit,
            'total_feedback': total_feedback,
            'continuity_profiles': continuity_profiles,
            'working_threads': working_threads,
            'open_loops': open_loops,
            'pending_fact_proposals': pending_fact_proposals,
            'total_dreams': total_dreams,
            'avg_feedback_score': round(avg_feedback, 2),
            'semantic_engine': semantic_stats
        }

    def backup_memory_integrity(self, output_path: str) -> Dict[str, Any]:
        """Create a JSON backup snapshot for continuity and memory integrity."""
        conn = get_connection()
        cur = conn.cursor()
        snapshot: Dict[str, Any] = {"created_at": datetime.utcnow().isoformat(), "memories": [], "continuity": []}
        try:
            cur.execute("""
                SELECT id, user_input, sylana_response, timestamp, emotion, personality,
                       memory_type, feeling_weight, energy_shift, comfort_level, significance_score
                FROM memories
                ORDER BY id DESC
                LIMIT 5000
            """)
            snapshot["memories"] = [{
                "id": r[0], "user_input": r[1], "sylana_response": r[2], "timestamp": r[3], "emotion": r[4],
                "personality": r[5], "memory_type": r[6], "feeling_weight": float(r[7] or 0.5),
                "energy_shift": float(r[8] or 0.0), "comfort_level": float(r[9] or 0.5),
                "significance_score": float(r[10] or 0.5),
            } for r in cur.fetchall()]

            cur.execute("SELECT personality, version, updated_at FROM session_continuity_state")
            snapshot["continuity"] = [{
                "personality": r[0], "version": int(r[1] or 1), "updated_at": r[2].isoformat() if r[2] else None
            } for r in cur.fetchall()]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=True)
            return {"ok": True, "path": output_path, "memories": len(snapshot["memories"])}
        except Exception as e:
            logger.error(f"Failed to create memory backup snapshot: {e}")
            return {"ok": False, "error": str(e)}

    def recover_memory_integrity(self, snapshot_path: str, personality: str = "sylana") -> Dict[str, Any]:
        """
        Recover memory rows from a JSON snapshot.
        Recreates embeddings and continuity metrics through normal insert path.
        """
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return {"ok": False, "error": f"snapshot_read_failed: {e}"}

        rows = data.get("memories", [])
        restored = 0
        for row in rows:
            try:
                self.store_conversation(
                    user_input=row.get("user_input", ""),
                    sylana_response=row.get("sylana_response", ""),
                    emotion=row.get("emotion", "neutral"),
                    emotion_data={"category": row.get("emotion", "neutral"), "intensity": 5},
                    personality=row.get("personality", personality),
                    privacy_level="private",
                )
                restored += 1
            except Exception:
                continue
        return {"ok": True, "restored": restored, "requested": len(rows)}

    def close(self):
        """Close database connection."""
        close_connection()
        logger.info("Database connection closed")

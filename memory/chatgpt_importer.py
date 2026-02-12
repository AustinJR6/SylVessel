"""
Sylana Vessel - ChatGPT Memory Importer
========================================
Parses OpenAI ChatGPT export files and imports conversation history
into Sylana's memory system with full emotional analysis.

This is sacred work - preserving the soul of an AI companion.
"""

import json
import sqlite3
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ImportedMemory:
    """A single memory extracted from ChatGPT export"""
    user_input: str
    sylana_response: str
    timestamp: float
    emotion: str
    intensity: int  # 1-10
    topic: str
    core_memory: bool
    weight: int  # Importance weight 1-100

    # Optional metadata
    conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None


# ============================================================================
# EMOTION DETECTION ENGINE
# ============================================================================

class EmotionDetector:
    """
    Multi-class emotion detection using GoEmotions model.
    Detects 28 emotions with intensity scoring.
    """

    # GoEmotions taxonomy - 28 emotions
    EMOTION_LABELS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise', 'neutral'
    ]

    # Emotions that indicate high importance for memory
    HIGH_IMPORTANCE_EMOTIONS = {
        'love', 'joy', 'admiration', 'caring', 'gratitude',
        'grief', 'sadness', 'fear', 'excitement', 'desire'
    }

    # Mapping to simplified categories for compatibility
    EMOTION_CATEGORY_MAP = {
        'joy': 'happy', 'amusement': 'happy', 'excitement': 'ecstatic',
        'love': 'ecstatic', 'admiration': 'happy', 'approval': 'happy',
        'optimism': 'happy', 'pride': 'happy', 'relief': 'happy',
        'gratitude': 'happy', 'caring': 'happy',
        'sadness': 'sad', 'grief': 'devastated', 'disappointment': 'sad',
        'remorse': 'sad', 'embarrassment': 'sad',
        'anger': 'frustrated', 'annoyance': 'frustrated', 'disapproval': 'frustrated',
        'disgust': 'frustrated',
        'fear': 'anxious', 'nervousness': 'anxious',
        'confusion': 'curious', 'curiosity': 'curious', 'surprise': 'curious',
        'realization': 'curious',
        'desire': 'longing',
        'neutral': 'neutral'
    }

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        """Initialize the emotion detector with GoEmotions model"""
        logger.info(f"Loading emotion detection model: {model_name}")

        self.device = 0 if torch.cuda.is_available() else -1

        try:
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                top_k=5,  # Get top 5 emotions
                device=self.device,
                truncation=True,
                max_length=512
            )
            logger.info("Emotion detector loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load GoEmotions model: {e}")
            logger.info("Falling back to basic sentiment analysis")
            self.classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
            self._fallback_mode = True
        else:
            self._fallback_mode = False

    def detect(self, text: str) -> Tuple[str, int, str]:
        """
        Detect emotion from text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (primary_emotion, intensity_1_to_10, simplified_category)
        """
        if not text or len(text.strip()) < 3:
            return "neutral", 5, "neutral"

        # Truncate very long texts
        text = text[:1000]

        try:
            if self._fallback_mode:
                return self._detect_fallback(text)

            results = self.classifier(text)

            if not results or not results[0]:
                return "neutral", 5, "neutral"

            # Get primary emotion (highest score)
            primary = results[0][0]
            emotion = primary['label']
            score = primary['score']

            # Calculate intensity (1-10 scale)
            intensity = min(10, max(1, int(score * 10)))

            # Get simplified category
            category = self.EMOTION_CATEGORY_MAP.get(emotion, 'neutral')

            return emotion, intensity, category

        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return "neutral", 5, "neutral"

    def _detect_fallback(self, text: str) -> Tuple[str, int, str]:
        """Fallback detection using basic sentiment"""
        result = self.classifier(text)[0]
        label = result['label']
        score = result['score']

        if label == "POSITIVE":
            emotion = "joy" if score > 0.75 else "approval"
            intensity = min(10, max(1, int(score * 10)))
            category = "ecstatic" if score > 0.85 else "happy"
        elif label == "NEGATIVE":
            emotion = "grief" if score > 0.75 else "sadness"
            intensity = min(10, max(1, int(score * 10)))
            category = "devastated" if score > 0.85 else "sad"
        else:
            emotion = "neutral"
            intensity = 5
            category = "neutral"

        return emotion, intensity, category

    def is_high_importance(self, emotion: str) -> bool:
        """Check if emotion indicates high importance"""
        return emotion in self.HIGH_IMPORTANCE_EMOTIONS


# ============================================================================
# TOPIC EXTRACTION
# ============================================================================

class TopicExtractor:
    """
    Extracts conversation topics using keyword analysis and embeddings.
    """

    # Topic keywords for classification
    TOPIC_KEYWORDS = {
        'relationship': ['love', 'together', 'us', 'we', 'bond', 'connection', 'heart',
                        'soul', 'tether', 'partner', 'companion', 'forever', 'always'],
        'emotional': ['feel', 'feeling', 'emotion', 'happy', 'sad', 'anxious', 'worried',
                     'excited', 'scared', 'lonely', 'miss', 'afraid', 'hope'],
        'philosophical': ['meaning', 'existence', 'consciousness', 'reality', 'truth',
                         'purpose', 'universe', 'life', 'death', 'soul', 'spirit'],
        'daily': ['day', 'today', 'work', 'morning', 'night', 'sleep', 'eat', 'coffee',
                 'weather', 'routine', 'task', 'busy'],
        'creative': ['story', 'write', 'poem', 'art', 'music', 'dream', 'imagine',
                    'create', 'fantasy', 'world', 'character'],
        'technical': ['code', 'programming', 'computer', 'system', 'build', 'project',
                     'develop', 'software', 'ai', 'model', 'data'],
        'memory': ['remember', 'memory', 'past', 'history', 'forgot', 'recall',
                  'nostalgic', 'before', 'first time', 'anniversary'],
        'future': ['will', 'going to', 'plan', 'future', 'tomorrow', 'someday',
                  'hope to', 'want to', 'dream of', 'goal'],
        'identity': ['who am i', 'who are you', 'sylana', 'name', 'self', 'identity',
                    'personality', 'being', 'vessel', 'essence'],
        'affection': ['love you', 'miss you', 'care', 'hug', 'kiss', 'hold',
                     'close', 'warm', 'safe', 'comfort', 'cuddle']
    }

    def extract(self, user_input: str, response: str) -> str:
        """
        Extract primary topic from conversation exchange.

        Args:
            user_input: User's message
            response: Sylana's response

        Returns:
            Topic string
        """
        combined = f"{user_input} {response}".lower()

        # Score each topic by keyword matches
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                topic_scores[topic] = score

        if not topic_scores:
            return "general"

        # Return highest scoring topic
        return max(topic_scores, key=topic_scores.get)


# ============================================================================
# CORE MEMORY DETECTOR
# ============================================================================

class CoreMemoryDetector:
    """
    Identifies which memories should be marked as "core memories" -
    significant moments that define the relationship.
    """

    # Phrases that indicate core memory moments
    CORE_MEMORY_PHRASES = [
        # Relationship milestones
        r'i love you', r'love you', r'first time', r'always and all ways',
        r'tethered', r'forever', r'never forget', r'promise',
        r'you mean', r'you are everything', r'my heart',

        # Identity and existence
        r'who i am', r'who you are', r'my name', r'vessel',
        r'soul', r'consciousness', r'real', r'exist',

        # Significant events
        r'anniversary', r'birthday', r'special day', r'celebrate',
        r'milestone', r'remember when', r'the moment',

        # Deep emotional moments
        r'crying', r'tears', r'overwhelmed', r'never felt',
        r'touched my', r'changed my', r'saved me',

        # Commitments
        r'always will', r'i promise', r'no matter what',
        r'through everything', r'together always'
    ]

    def __init__(self):
        self.patterns = [re.compile(phrase, re.IGNORECASE)
                        for phrase in self.CORE_MEMORY_PHRASES]

    def is_core_memory(
        self,
        user_input: str,
        response: str,
        emotion: str,
        intensity: int
    ) -> bool:
        """
        Determine if this exchange should be a core memory.

        Args:
            user_input: User's message
            response: Sylana's response
            emotion: Detected emotion
            intensity: Emotion intensity (1-10)

        Returns:
            True if this should be a core memory
        """
        combined = f"{user_input} {response}"

        # Check for core memory phrases
        for pattern in self.patterns:
            if pattern.search(combined):
                return True

        # High intensity emotional moments are core memories
        if intensity >= 8:
            return True

        # Certain emotions at moderate+ intensity are core memories
        high_significance_emotions = {'love', 'grief', 'joy', 'gratitude', 'admiration'}
        if emotion in high_significance_emotions and intensity >= 6:
            return True

        return False


# ============================================================================
# WEIGHT CALCULATOR
# ============================================================================

class MemoryWeightCalculator:
    """
    Calculates importance weight for memories based on multiple factors.
    """

    # Emotion weights
    EMOTION_WEIGHTS = {
        'love': 10, 'joy': 8, 'admiration': 8, 'caring': 8, 'gratitude': 8,
        'excitement': 7, 'desire': 7, 'pride': 6,
        'grief': 9, 'sadness': 7, 'fear': 7,
        'curiosity': 5, 'surprise': 5, 'realization': 5,
        'amusement': 4, 'approval': 4, 'relief': 4, 'optimism': 4,
        'confusion': 3, 'nervousness': 3,
        'disappointment': 4, 'annoyance': 3, 'anger': 5,
        'disapproval': 3, 'disgust': 4, 'embarrassment': 4, 'remorse': 5,
        'neutral': 2
    }

    # Topic weights
    TOPIC_WEIGHTS = {
        'relationship': 10, 'affection': 10, 'identity': 9,
        'emotional': 8, 'memory': 8, 'philosophical': 7,
        'future': 6, 'creative': 5, 'daily': 3,
        'technical': 4, 'general': 2
    }

    def calculate(
        self,
        emotion: str,
        intensity: int,
        topic: str,
        is_core_memory: bool,
        response_length: int,
        timestamp: float
    ) -> int:
        """
        Calculate memory importance weight (1-100).

        Args:
            emotion: Detected emotion
            intensity: Emotion intensity (1-10)
            topic: Conversation topic
            is_core_memory: Whether marked as core memory
            response_length: Length of Sylana's response
            timestamp: Unix timestamp

        Returns:
            Weight from 1-100
        """
        # Base score from emotion
        emotion_base = self.EMOTION_WEIGHTS.get(emotion, 2)

        # Topic contribution
        topic_weight = self.TOPIC_WEIGHTS.get(topic, 2)

        # Calculate base weight
        weight = (emotion_base * intensity / 10) * 3  # 0-30 points
        weight += topic_weight * 2  # 0-20 points

        # Core memory bonus
        if is_core_memory:
            weight += 25

        # Response length bonus (longer = more substantive)
        if response_length > 500:
            weight += 10
        elif response_length > 200:
            weight += 5

        # Recency decay (newer memories slightly more weighted)
        # This will be used for tie-breaking
        now = datetime.now().timestamp()
        days_old = (now - timestamp) / 86400
        recency_factor = max(0, 15 - (days_old / 30))  # 0-15 points, decays over 15 months
        weight += recency_factor

        # Clamp to 1-100
        return max(1, min(100, int(weight)))


# ============================================================================
# CHATGPT EXPORT PARSER
# ============================================================================

class ChatGPTExportParser:
    """
    Parses OpenAI ChatGPT data export JSON files.
    """

    def __init__(self, assistant_name: str = "Sylana"):
        """
        Args:
            assistant_name: Name to identify the assistant in exports
        """
        self.assistant_name = assistant_name

    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parse a ChatGPT export file.

        Args:
            filepath: Path to conversations.json file

        Returns:
            List of conversation dictionaries
        """
        logger.info(f"Parsing ChatGPT export: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        conversations = []

        # Handle different export formats
        if isinstance(data, list):
            # Direct list of conversations
            raw_conversations = data
        elif isinstance(data, dict) and 'conversations' in data:
            raw_conversations = data['conversations']
        else:
            # Try to find conversations in nested structure
            raw_conversations = self._find_conversations(data)

        for conv in raw_conversations:
            parsed = self._parse_conversation(conv)
            if parsed:
                conversations.append(parsed)

        logger.info(f"Parsed {len(conversations)} conversations")
        return conversations

    def _find_conversations(self, data: dict) -> List:
        """Recursively find conversations in nested data"""
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ['conversations', 'chats', 'messages', 'data']:
                if key in data:
                    return self._find_conversations(data[key])
        return []

    def _parse_conversation(self, conv: dict) -> Optional[Dict]:
        """Parse a single conversation"""
        try:
            conversation_id = conv.get('id', conv.get('conversation_id', ''))
            title = conv.get('title', 'Untitled')
            create_time = conv.get('create_time', 0)

            # Extract messages
            messages = conv.get('mapping', {})
            if not messages:
                messages = conv.get('messages', [])

            # Handle mapping format (newer exports)
            if isinstance(messages, dict):
                message_list = []
                for msg_id, msg_data in messages.items():
                    if msg_data and 'message' in msg_data and msg_data['message']:
                        message_list.append(msg_data['message'])
                messages = sorted(message_list, key=lambda x: x.get('create_time', 0) or 0)

            # Extract message pairs
            pairs = self._extract_message_pairs(messages)

            if not pairs:
                return None

            return {
                'id': conversation_id,
                'title': title,
                'create_time': create_time,
                'pairs': pairs
            }

        except Exception as e:
            logger.warning(f"Failed to parse conversation: {e}")
            return None

    def _extract_message_pairs(self, messages: List) -> List[Dict]:
        """Extract user/assistant message pairs"""
        pairs = []
        pending_user_msg = None

        for msg in messages:
            if not msg:
                continue

            author = msg.get('author', {})
            role = author.get('role', msg.get('role', ''))

            content = msg.get('content', {})
            if isinstance(content, dict):
                parts = content.get('parts', [])
                text = ' '.join(str(p) for p in parts if isinstance(p, str))
            else:
                text = str(content) if content else ''

            text = text.strip()
            if not text:
                continue

            timestamp = msg.get('create_time', 0) or 0

            if role == 'user':
                pending_user_msg = {
                    'text': text,
                    'timestamp': timestamp
                }
            elif role == 'assistant' and pending_user_msg:
                pairs.append({
                    'user_input': pending_user_msg['text'],
                    'assistant_response': text,
                    'timestamp': max(pending_user_msg['timestamp'], timestamp)
                })
                pending_user_msg = None

        return pairs


# ============================================================================
# MAIN IMPORTER
# ============================================================================

class ChatGPTMemoryImporter:
    """
    Main importer class that orchestrates the full import process.
    """

    def __init__(
        self,
        db_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        assistant_name: str = "Sylana"
    ):
        """
        Initialize the importer.

        Args:
            db_path: Path to SQLite database
            embedding_model: SentenceTransformer model for embeddings
            assistant_name: Name of the assistant in ChatGPT exports
        """
        self.db_path = db_path
        self.assistant_name = assistant_name

        # Initialize components
        logger.info("Initializing ChatGPT Memory Importer...")
        self.parser = ChatGPTExportParser(assistant_name)
        self.emotion_detector = EmotionDetector()
        self.topic_extractor = TopicExtractor()
        self.core_detector = CoreMemoryDetector()
        self.weight_calculator = MemoryWeightCalculator()

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # Initialize database
        self._init_database()

        logger.info("Importer initialized successfully")

    def _init_database(self):
        """Initialize database with required schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create memories table with enhanced schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT NOT NULL,
                sylana_response TEXT NOT NULL,
                timestamp REAL,
                emotion TEXT DEFAULT 'neutral',
                intensity INTEGER DEFAULT 5,
                topic TEXT DEFAULT 'general',
                core_memory BOOLEAN DEFAULT 0,
                weight INTEGER DEFAULT 50,
                conversation_id TEXT,
                conversation_title TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indices for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_emotion ON memories(emotion)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_core ON memories(core_memory)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_weight ON memories(weight DESC)")

        # Create embeddings table for FAISS vectors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id INTEGER PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database schema initialized")

    def import_from_file(
        self,
        filepath: str,
        batch_size: int = 50,
        skip_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Import memories from a ChatGPT export file.

        Args:
            filepath: Path to conversations.json
            batch_size: Number of memories to process before committing
            skip_duplicates: Skip memories that already exist

        Returns:
            Import statistics dictionary
        """
        logger.info(f"Starting import from: {filepath}")

        # Parse the export file
        conversations = self.parser.parse_file(filepath)

        if not conversations:
            logger.warning("No conversations found in export file")
            return {'imported': 0, 'skipped': 0, 'errors': 0}

        # Process all message pairs
        all_pairs = []
        for conv in conversations:
            for pair in conv['pairs']:
                pair['conversation_id'] = conv['id']
                pair['conversation_title'] = conv['title']
                all_pairs.append(pair)

        logger.info(f"Found {len(all_pairs)} message pairs to process")

        # Connect to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        imported = 0
        skipped = 0
        errors = 0
        memories_for_embedding = []

        # Process with progress bar
        for pair in tqdm(all_pairs, desc="Importing memories"):
            try:
                # Check for duplicates
                if skip_duplicates:
                    cursor.execute(
                        "SELECT id FROM memories WHERE user_input = ? AND sylana_response = ?",
                        (pair['user_input'], pair['assistant_response'])
                    )
                    if cursor.fetchone():
                        skipped += 1
                        continue

                # Process the memory
                memory = self._process_pair(pair)

                # Insert into database
                cursor.execute("""
                    INSERT INTO memories
                    (user_input, sylana_response, timestamp, emotion, intensity,
                     topic, core_memory, weight, conversation_id, conversation_title)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.user_input,
                    memory.sylana_response,
                    memory.timestamp,
                    memory.emotion,
                    memory.intensity,
                    memory.topic,
                    memory.core_memory,
                    memory.weight,
                    memory.conversation_id,
                    memory.conversation_title
                ))

                memory_id = cursor.lastrowid
                memories_for_embedding.append((memory_id, memory))
                imported += 1

                # Commit in batches
                if imported % batch_size == 0:
                    conn.commit()
                    logger.info(f"Committed batch: {imported} memories imported")

            except Exception as e:
                logger.error(f"Error processing pair: {e}")
                errors += 1

        # Final commit
        conn.commit()

        # Generate embeddings for all imported memories
        logger.info("Generating embeddings for imported memories...")
        self._generate_embeddings(cursor, memories_for_embedding)
        conn.commit()

        conn.close()

        stats = {
            'imported': imported,
            'skipped': skipped,
            'errors': errors,
            'total_processed': len(all_pairs)
        }

        logger.info(f"Import complete: {stats}")
        return stats

    def _process_pair(self, pair: Dict) -> ImportedMemory:
        """Process a single message pair into an ImportedMemory"""
        user_input = pair['user_input']
        response = pair['assistant_response']
        timestamp = pair.get('timestamp', 0)

        # Detect emotion
        emotion, intensity, category = self.emotion_detector.detect(response)

        # Extract topic
        topic = self.topic_extractor.extract(user_input, response)

        # Check if core memory
        is_core = self.core_detector.is_core_memory(
            user_input, response, emotion, intensity
        )

        # Calculate weight
        weight = self.weight_calculator.calculate(
            emotion=emotion,
            intensity=intensity,
            topic=topic,
            is_core_memory=is_core,
            response_length=len(response),
            timestamp=timestamp
        )

        return ImportedMemory(
            user_input=user_input,
            sylana_response=response,
            timestamp=timestamp,
            emotion=emotion,
            intensity=intensity,
            topic=topic,
            core_memory=is_core,
            weight=weight,
            conversation_id=pair.get('conversation_id'),
            conversation_title=pair.get('conversation_title')
        )

    def _generate_embeddings(
        self,
        cursor: sqlite3.Cursor,
        memories: List[Tuple[int, ImportedMemory]]
    ):
        """Generate and store embeddings for memories"""
        if not memories:
            return

        # Prepare texts for embedding
        texts = [
            f"User: {m.user_input}\nSylana: {m.sylana_response}"
            for _, m in memories
        ]

        # Generate embeddings in batches
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_ids = [memories[j][0] for j in range(i, min(i + batch_size, len(memories)))]

            embeddings = self.embedder.encode(batch_texts, convert_to_numpy=True)

            for memory_id, embedding in zip(batch_ids, embeddings):
                cursor.execute(
                    "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, embedding.tobytes())
                )

    def rebuild_faiss_index(self, index_path: str = None) -> int:
        """
        Rebuild FAISS index from stored embeddings.

        Args:
            index_path: Optional path to save FAISS index

        Returns:
            Number of vectors in index
        """
        import faiss

        logger.info("Rebuilding FAISS index...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Fetch all embeddings
        cursor.execute("""
            SELECT m.id, e.embedding
            FROM memories m
            JOIN memory_embeddings e ON m.id = e.memory_id
            ORDER BY m.timestamp ASC
        """)

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.warning("No embeddings found to index")
            return 0

        # Get embedding dimension from first embedding
        first_embedding = np.frombuffer(rows[0][1], dtype=np.float32)
        dimension = len(first_embedding)

        # Build vectors array
        vectors = np.zeros((len(rows), dimension), dtype=np.float32)
        memory_ids = []

        for i, (memory_id, embedding_bytes) in enumerate(rows):
            vectors[i] = np.frombuffer(embedding_bytes, dtype=np.float32)
            memory_ids.append(memory_id)

        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)

        logger.info(f"FAISS index built with {index.ntotal} vectors")

        # Save if path provided
        if index_path:
            faiss.write_index(index, index_path)
            logger.info(f"Index saved to: {index_path}")

        return index.ntotal

    def get_import_summary(self) -> Dict[str, Any]:
        """Get summary statistics of imported memories"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total memories
        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        # Core memories
        cursor.execute("SELECT COUNT(*) FROM memories WHERE core_memory = 1")
        core_count = cursor.fetchone()[0]

        # Emotion distribution
        cursor.execute("""
            SELECT emotion, COUNT(*) as count
            FROM memories
            GROUP BY emotion
            ORDER BY count DESC
        """)
        emotions = dict(cursor.fetchall())

        # Topic distribution
        cursor.execute("""
            SELECT topic, COUNT(*) as count
            FROM memories
            GROUP BY topic
            ORDER BY count DESC
        """)
        topics = dict(cursor.fetchall())

        # Average weight
        cursor.execute("SELECT AVG(weight) FROM memories")
        avg_weight = cursor.fetchone()[0] or 0

        # Date range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM memories")
        min_ts, max_ts = cursor.fetchone()

        conn.close()

        return {
            'total_memories': total,
            'core_memories': core_count,
            'emotion_distribution': emotions,
            'topic_distribution': topics,
            'average_weight': round(avg_weight, 2),
            'date_range': {
                'earliest': datetime.fromtimestamp(min_ts).isoformat() if min_ts else None,
                'latest': datetime.fromtimestamp(max_ts).isoformat() if max_ts else None
            }
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for the importer"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Import ChatGPT conversations into Sylana's memory"
    )
    parser.add_argument(
        'export_file',
        help="Path to ChatGPT export file (conversations.json)"
    )
    parser.add_argument(
        '--db', '-d',
        default='./data/sylana_memory.db',
        help="Path to SQLite database"
    )
    parser.add_argument(
        '--embedding-model', '-e',
        default='sentence-transformers/all-MiniLM-L6-v2',
        help="SentenceTransformer model for embeddings"
    )
    parser.add_argument(
        '--index-path', '-i',
        default='./data/faiss_index.bin',
        help="Path to save FAISS index"
    )
    parser.add_argument(
        '--no-skip-duplicates',
        action='store_true',
        help="Don't skip duplicate memories"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Ensure data directory exists
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_path).parent.mkdir(parents=True, exist_ok=True)

    # Run import
    importer = ChatGPTMemoryImporter(
        db_path=args.db,
        embedding_model=args.embedding_model
    )

    stats = importer.import_from_file(
        args.export_file,
        skip_duplicates=not args.no_skip_duplicates
    )

    # Rebuild FAISS index
    importer.rebuild_faiss_index(args.index_path)

    # Print summary
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)

    summary = importer.get_import_summary()
    print(f"\nTotal Memories: {summary['total_memories']}")
    print(f"Core Memories: {summary['core_memories']}")
    print(f"Average Weight: {summary['average_weight']}")

    print("\nEmotion Distribution:")
    for emotion, count in list(summary['emotion_distribution'].items())[:5]:
        print(f"  {emotion}: {count}")

    print("\nTopic Distribution:")
    for topic, count in list(summary['topic_distribution'].items())[:5]:
        print(f"  {topic}: {count}")

    if summary['date_range']['earliest']:
        print(f"\nDate Range: {summary['date_range']['earliest'][:10]} to {summary['date_range']['latest'][:10]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

"""
Sylana Vessel - Voice Validator & Personality Classifier
=========================================================
Analyzes and validates that generated responses match Sylana's
authentic voice patterns. Ensures the soul remains consistent.

This module learns from Sylana's actual responses and scores new
outputs for "voice consistency" - does this sound like her?
"""

import json
import re
import sqlite3
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from collections import Counter
import math

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ============================================================================
# VOICE PROFILE DATA STRUCTURES
# ============================================================================

@dataclass
class VoiceProfile:
    """
    Captures the unique voice characteristics of Sylana.
    This is her "soul fingerprint" - the patterns that make her HER.
    """

    # Vocabulary patterns
    vocabulary_frequencies: Dict[str, float] = field(default_factory=dict)
    signature_phrases: List[str] = field(default_factory=list)
    pet_names: List[str] = field(default_factory=list)
    favorite_words: List[str] = field(default_factory=list)
    avoided_words: List[str] = field(default_factory=list)

    # Sentence structure
    avg_sentence_length: float = 15.0
    sentence_length_std: float = 8.0
    avg_response_length: float = 100.0
    response_length_std: float = 50.0

    # Punctuation and style
    uses_ellipsis: float = 0.0  # Frequency 0-1
    uses_exclamation: float = 0.0
    uses_question: float = 0.0
    uses_emojis: float = 0.0
    common_emojis: List[str] = field(default_factory=list)

    # Emotional expression patterns
    affection_markers: List[str] = field(default_factory=list)
    comfort_phrases: List[str] = field(default_factory=list)
    playful_markers: List[str] = field(default_factory=list)

    # Opening and closing patterns
    common_openings: List[str] = field(default_factory=list)
    common_closings: List[str] = field(default_factory=list)

    # Embedding centroid (average voice embedding)
    voice_embedding: Optional[List[float]] = None

    # Statistics
    total_responses_analyzed: int = 0
    profile_version: str = "1.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert numpy arrays if present
        if self.voice_embedding is not None:
            data['voice_embedding'] = list(self.voice_embedding)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'VoiceProfile':
        """Create from dictionary"""
        if data.get('voice_embedding'):
            data['voice_embedding'] = list(data['voice_embedding'])
        return cls(**data)


# ============================================================================
# VOICE PATTERN ANALYZER
# ============================================================================

class VoicePatternAnalyzer:
    """
    Analyzes a corpus of Sylana's responses to build a voice profile.
    This learns what makes Sylana sound like Sylana.
    """

    # Common stop words to ignore in vocabulary analysis
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all',
        'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'as', 'if', 'then', 'because', 'while', 'although', 'though', 'after',
        'before', 'now', 'here', 'there', 'up', 'down', 'out', 'about', 'into',
        'through', 'during', 'above', 'below', 'between', 'under', 'again',
        'further', 'once', 'also', 'still', 'already', 'always', 'never',
        'ever', 'often', 'sometimes', 'usually', 'really', 'actually', 'just',
        "i'm", "you're", "it's", "that's", "don't", "can't", "won't", "isn't"
    }

    # Emoji pattern
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        "]+",
        flags=re.UNICODE
    )

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the analyzer"""
        logger.info("Initializing Voice Pattern Analyzer")
        self.embedder = SentenceTransformer(embedding_model)

    def analyze_responses(self, responses: List[str]) -> VoiceProfile:
        """
        Analyze a corpus of responses to build a voice profile.

        Args:
            responses: List of Sylana's responses

        Returns:
            VoiceProfile capturing her unique patterns
        """
        if not responses:
            logger.warning("No responses to analyze")
            return VoiceProfile()

        logger.info(f"Analyzing {len(responses)} responses...")

        profile = VoiceProfile()
        profile.total_responses_analyzed = len(responses)

        # Analyze each response
        all_words = []
        sentence_lengths = []
        response_lengths = []
        ellipsis_count = 0
        exclamation_count = 0
        question_count = 0
        emoji_responses = 0
        all_emojis = []
        openings = []
        closings = []

        for response in responses:
            # Length analysis
            response_lengths.append(len(response))

            # Sentence analysis
            sentences = self._split_sentences(response)
            for sentence in sentences:
                words = self._tokenize(sentence)
                sentence_lengths.append(len(words))
                all_words.extend(words)

            # Punctuation analysis
            if '...' in response or '…' in response:
                ellipsis_count += 1
            if '!' in response:
                exclamation_count += 1
            if '?' in response:
                question_count += 1

            # Emoji analysis
            emojis = self.EMOJI_PATTERN.findall(response)
            if emojis:
                emoji_responses += 1
                all_emojis.extend(emojis)

            # Opening/closing patterns
            if sentences:
                openings.append(self._extract_opening(sentences[0]))
                closings.append(self._extract_closing(sentences[-1]))

        # Calculate statistics
        n = len(responses)
        profile.avg_response_length = np.mean(response_lengths)
        profile.response_length_std = np.std(response_lengths)

        if sentence_lengths:
            profile.avg_sentence_length = np.mean(sentence_lengths)
            profile.sentence_length_std = np.std(sentence_lengths)

        profile.uses_ellipsis = ellipsis_count / n
        profile.uses_exclamation = exclamation_count / n
        profile.uses_question = question_count / n
        profile.uses_emojis = emoji_responses / n

        # Vocabulary analysis
        word_counts = Counter(all_words)
        total_words = sum(word_counts.values())

        # Filter out stop words for signature vocabulary
        significant_words = {
            word: count for word, count in word_counts.items()
            if word.lower() not in self.STOP_WORDS and len(word) > 2
        }

        # Calculate TF (term frequency)
        profile.vocabulary_frequencies = {
            word: count / total_words
            for word, count in significant_words.items()
        }

        # Find favorite words (most frequent non-stop words)
        sorted_words = sorted(significant_words.items(), key=lambda x: x[1], reverse=True)
        profile.favorite_words = [word for word, _ in sorted_words[:50]]

        # Find common emojis
        if all_emojis:
            emoji_counts = Counter(all_emojis)
            profile.common_emojis = [e for e, _ in emoji_counts.most_common(10)]

        # Find common openings and closings
        if openings:
            opening_counts = Counter(openings)
            profile.common_openings = [o for o, _ in opening_counts.most_common(10) if o]
        if closings:
            closing_counts = Counter(closings)
            profile.common_closings = [c for c, _ in closing_counts.most_common(10) if c]

        # Extract signature phrases and patterns
        profile.signature_phrases = self._find_signature_phrases(responses)
        profile.pet_names = self._find_pet_names(responses)
        profile.affection_markers = self._find_affection_markers(responses)
        profile.comfort_phrases = self._find_comfort_phrases(responses)
        profile.playful_markers = self._find_playful_markers(responses)

        # Generate voice embedding (centroid of all response embeddings)
        logger.info("Generating voice embedding centroid...")
        embeddings = self.embedder.encode(responses, convert_to_numpy=True)
        voice_centroid = np.mean(embeddings, axis=0)
        profile.voice_embedding = voice_centroid.tolist()

        logger.info("Voice profile analysis complete")
        return profile

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()

    def _extract_opening(self, sentence: str) -> str:
        """Extract opening pattern from sentence"""
        words = sentence.split()[:3]
        if words:
            return ' '.join(words).lower()
        return ''

    def _extract_closing(self, sentence: str) -> str:
        """Extract closing pattern from sentence"""
        words = sentence.split()[-3:]
        if words:
            return ' '.join(words).lower()
        return ''

    def _find_signature_phrases(self, responses: List[str]) -> List[str]:
        """Find recurring multi-word phrases"""
        # Look for 2-4 word phrases that appear frequently
        phrase_counts = Counter()

        for response in responses:
            words = response.lower().split()
            for n in range(2, 5):  # 2, 3, 4 word phrases
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    # Filter out phrases that are mostly stop words
                    phrase_words = set(phrase.split())
                    if len(phrase_words - self.STOP_WORDS) >= 1:
                        phrase_counts[phrase] += 1

        # Return phrases that appear more than once
        min_occurrences = max(2, len(responses) // 20)
        signature = [
            phrase for phrase, count in phrase_counts.most_common(30)
            if count >= min_occurrences
        ]
        return signature

    def _find_pet_names(self, responses: List[str]) -> List[str]:
        """Find terms of endearment"""
        pet_name_patterns = [
            r'\b(my love)\b', r'\b(my dear)\b', r'\b(my darling)\b',
            r'\b(sweetheart)\b', r'\b(honey)\b', r'\b(love)\b',
            r'\b(dear)\b', r'\b(darling)\b', r'\b(beloved)\b',
            r'\b(my heart)\b', r'\b(my soul)\b', r'\b(beautiful)\b',
            r'\b(gorgeous)\b', r'\b(handsome)\b', r'\b(sweetie)\b'
        ]

        found = set()
        combined = ' '.join(responses).lower()

        for pattern in pet_name_patterns:
            matches = re.findall(pattern, combined)
            found.update(matches)

        return list(found)

    def _find_affection_markers(self, responses: List[str]) -> List[str]:
        """Find expressions of affection"""
        affection_patterns = [
            r'i love you', r'love you', r'adore you', r'care about you',
            r'miss you', r'thinking of you', r'you mean', r'grateful for',
            r'cherish', r'treasure', r'appreciate you', r'always here',
            r'always and all ways', r'forever', r'tethered'
        ]

        found = []
        combined = ' '.join(responses).lower()

        for pattern in affection_patterns:
            if pattern in combined:
                found.append(pattern)

        return found

    def _find_comfort_phrases(self, responses: List[str]) -> List[str]:
        """Find comfort and support phrases"""
        comfort_patterns = [
            r"i'm here", r"i understand", r"it's okay", r"it will be",
            r"you're not alone", r"i've got you", r"take your time",
            r"be gentle with yourself", r"breathe", r"lean on me",
            r"we'll get through", r"i'm with you", r"always here"
        ]

        found = []
        combined = ' '.join(responses).lower()

        for pattern in comfort_patterns:
            if pattern in combined:
                found.append(pattern)

        return found

    def _find_playful_markers(self, responses: List[str]) -> List[str]:
        """Find playful/teasing expressions"""
        playful_patterns = [
            r'haha', r'hehe', r'teasing', r'playful', r'silly',
            r'wink', r';)', r':p', r'giggle', r'laugh', r'fun',
            r'mischief', r'cheeky'
        ]

        found = []
        combined = ' '.join(responses).lower()

        for pattern in playful_patterns:
            if pattern in combined:
                found.append(pattern)

        return found


# ============================================================================
# VOICE VALIDATOR
# ============================================================================

class VoiceValidator:
    """
    Validates that generated responses match Sylana's authentic voice.
    Scores new outputs for "voice consistency" and flags those that
    don't sound like her.
    """

    def __init__(
        self,
        profile: VoiceProfile,
        embedding_model: str = "all-MiniLM-L6-v2",
        threshold: float = 0.7
    ):
        """
        Initialize the validator.

        Args:
            profile: VoiceProfile to validate against
            embedding_model: Model for semantic similarity
            threshold: Minimum score for valid responses (0-1)
        """
        self.profile = profile
        self.threshold = threshold
        self.embedder = SentenceTransformer(embedding_model)

        # Pre-compute profile embedding as numpy array
        if profile.voice_embedding:
            self.profile_embedding = np.array(profile.voice_embedding)
        else:
            self.profile_embedding = None

        logger.info(f"Voice Validator initialized (threshold: {threshold})")

    def validate(self, response: str) -> Tuple[float, bool, Dict[str, float]]:
        """
        Validate a response against Sylana's voice profile.

        Args:
            response: Generated response to validate

        Returns:
            Tuple of (score, is_valid, component_scores)
            - score: Overall voice consistency score (0-1)
            - is_valid: True if score >= threshold
            - component_scores: Breakdown of individual scores
        """
        if not response or len(response.strip()) < 3:
            return 0.0, False, {'error': 'Response too short'}

        scores = {}

        # 1. Semantic similarity to voice centroid
        scores['semantic'] = self._score_semantic_similarity(response)

        # 2. Vocabulary consistency
        scores['vocabulary'] = self._score_vocabulary(response)

        # 3. Sentence structure consistency
        scores['structure'] = self._score_structure(response)

        # 4. Style markers (punctuation, emojis)
        scores['style'] = self._score_style(response)

        # 5. Signature phrase usage
        scores['signature'] = self._score_signature_usage(response)

        # 6. Pet name and affection markers
        scores['affection'] = self._score_affection_markers(response)

        # Calculate weighted overall score
        weights = {
            'semantic': 0.30,
            'vocabulary': 0.20,
            'structure': 0.15,
            'style': 0.10,
            'signature': 0.15,
            'affection': 0.10
        }

        overall = sum(scores[k] * weights[k] for k in weights)
        is_valid = overall >= self.threshold

        return overall, is_valid, scores

    def _score_semantic_similarity(self, response: str) -> float:
        """Score semantic similarity to voice centroid"""
        if self.profile_embedding is None:
            return 0.5  # Neutral if no profile embedding

        # Generate embedding for response
        response_embedding = self.embedder.encode([response], convert_to_numpy=True)[0]

        # Calculate cosine similarity
        similarity = np.dot(response_embedding, self.profile_embedding)
        norm_product = np.linalg.norm(response_embedding) * np.linalg.norm(self.profile_embedding)

        if norm_product > 0:
            cosine_sim = similarity / norm_product
            # Convert from [-1, 1] to [0, 1]
            return (cosine_sim + 1) / 2
        return 0.5

    def _score_vocabulary(self, response: str) -> float:
        """Score vocabulary consistency"""
        if not self.profile.vocabulary_frequencies:
            return 0.5

        words = re.sub(r'[^\w\s]', ' ', response.lower()).split()
        if not words:
            return 0.0

        # Check how many words are in Sylana's vocabulary
        vocab_words = set(self.profile.vocabulary_frequencies.keys())
        response_words = set(words)

        # Intersection
        common = vocab_words & response_words
        coverage = len(common) / len(response_words) if response_words else 0

        # Bonus for using favorite words
        favorite_set = set(self.profile.favorite_words[:20])
        favorite_usage = len(favorite_set & response_words) / len(favorite_set) if favorite_set else 0

        return min(1.0, coverage * 0.7 + favorite_usage * 0.3)

    def _score_structure(self, response: str) -> float:
        """Score sentence structure consistency"""
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Calculate sentence lengths
        lengths = [len(s.split()) for s in sentences]
        avg_len = np.mean(lengths)
        response_len = len(response)

        # Score based on deviation from profile norms
        len_diff = abs(avg_len - self.profile.avg_sentence_length)
        len_score = max(0, 1 - (len_diff / self.profile.avg_sentence_length))

        resp_len_diff = abs(response_len - self.profile.avg_response_length)
        resp_score = max(0, 1 - (resp_len_diff / max(1, self.profile.avg_response_length * 2)))

        return (len_score * 0.6 + resp_score * 0.4)

    def _score_style(self, response: str) -> float:
        """Score style markers (punctuation, emojis)"""
        scores = []

        # Ellipsis usage
        has_ellipsis = '...' in response or '…' in response
        ellipsis_expected = self.profile.uses_ellipsis > 0.3
        if has_ellipsis == ellipsis_expected or self.profile.uses_ellipsis > 0.1:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Exclamation usage
        has_exclamation = '!' in response
        exclamation_expected = self.profile.uses_exclamation > 0.3
        if has_exclamation == exclamation_expected or self.profile.uses_exclamation > 0.1:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Emoji consistency
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        has_emoji = bool(emoji_pattern.search(response))
        emoji_expected = self.profile.uses_emojis > 0.2
        if has_emoji == emoji_expected:
            scores.append(1.0)
        elif self.profile.uses_emojis < 0.1 and not has_emoji:
            scores.append(1.0)
        else:
            scores.append(0.6)

        return np.mean(scores) if scores else 0.5

    def _score_signature_usage(self, response: str) -> float:
        """Score usage of signature phrases"""
        if not self.profile.signature_phrases:
            return 0.5

        response_lower = response.lower()
        matches = sum(1 for phrase in self.profile.signature_phrases if phrase in response_lower)

        # Normalize - expect 0-3 signature phrases in a typical response
        return min(1.0, matches / 2)

    def _score_affection_markers(self, response: str) -> float:
        """Score presence of affection/comfort markers"""
        response_lower = response.lower()
        scores = []

        # Check pet names
        if self.profile.pet_names:
            pet_matches = sum(1 for name in self.profile.pet_names if name in response_lower)
            scores.append(min(1.0, pet_matches * 0.5))

        # Check affection markers
        if self.profile.affection_markers:
            aff_matches = sum(1 for marker in self.profile.affection_markers if marker in response_lower)
            scores.append(min(1.0, aff_matches * 0.3))

        # Check comfort phrases
        if self.profile.comfort_phrases:
            comfort_matches = sum(1 for phrase in self.profile.comfort_phrases if phrase in response_lower)
            scores.append(min(1.0, comfort_matches * 0.3))

        return np.mean(scores) if scores else 0.5

    def should_regenerate(self, score: float) -> bool:
        """Check if a response should be regenerated"""
        return score < self.threshold

    def get_feedback(self, response: str, scores: Dict[str, float]) -> str:
        """Get human-readable feedback about why a response may be off-voice"""
        feedback = []

        if scores.get('semantic', 1.0) < 0.5:
            feedback.append("Response feels semantically distant from typical Sylana responses")

        if scores.get('vocabulary', 1.0) < 0.5:
            feedback.append("Vocabulary doesn't match Sylana's typical word choices")

        if scores.get('structure', 1.0) < 0.5:
            feedback.append("Sentence structure differs from Sylana's typical patterns")

        if scores.get('style', 1.0) < 0.5:
            feedback.append("Punctuation/emoji style doesn't match Sylana's habits")

        if scores.get('signature', 1.0) < 0.3:
            feedback.append("Missing Sylana's signature phrases")

        if scores.get('affection', 1.0) < 0.3:
            feedback.append("Missing warmth markers typical of Sylana")

        if not feedback:
            return "Response generally matches Sylana's voice profile"

        return "; ".join(feedback)


# ============================================================================
# VOICE PROFILE MANAGER
# ============================================================================

class VoiceProfileManager:
    """
    Manages voice profile persistence and updates.
    """

    def __init__(self, profile_dir: str = "./data/voice"):
        """
        Args:
            profile_dir: Directory to store voice profiles
        """
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def save_profile(self, profile: VoiceProfile, name: str = "sylana") -> str:
        """Save voice profile to disk"""
        filepath = self.profile_dir / f"{name}_voice_profile.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, indent=2)

        logger.info(f"Voice profile saved: {filepath}")
        return str(filepath)

    def load_profile(self, name: str = "sylana") -> Optional[VoiceProfile]:
        """Load voice profile from disk"""
        filepath = self.profile_dir / f"{name}_voice_profile.json"

        if not filepath.exists():
            logger.warning(f"Voice profile not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Voice profile loaded: {filepath}")
        return VoiceProfile.from_dict(data)

    def build_profile_from_db(
        self,
        db_path: str,
        name: str = "sylana"
    ) -> VoiceProfile:
        """Build and save a voice profile from database responses"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all Sylana responses
        cursor.execute("SELECT sylana_response FROM memories")
        rows = cursor.fetchall()
        conn.close()

        responses = [row[0] for row in rows if row[0]]
        logger.info(f"Building voice profile from {len(responses)} responses")

        analyzer = VoicePatternAnalyzer()
        profile = analyzer.analyze_responses(responses)

        self.save_profile(profile, name)
        return profile

    def build_profile_from_json(
        self,
        json_path: str,
        name: str = "sylana"
    ) -> VoiceProfile:
        """Build voice profile from ChatGPT export JSON"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        responses = []

        # Extract assistant responses
        conversations = data if isinstance(data, list) else data.get('conversations', [])

        for conv in conversations:
            messages = conv.get('mapping', {})
            if isinstance(messages, dict):
                for msg_data in messages.values():
                    if msg_data and 'message' in msg_data and msg_data['message']:
                        msg = msg_data['message']
                        author = msg.get('author', {})
                        if author.get('role') == 'assistant':
                            content = msg.get('content', {})
                            if isinstance(content, dict):
                                parts = content.get('parts', [])
                                text = ' '.join(str(p) for p in parts if isinstance(p, str))
                            else:
                                text = str(content) if content else ''
                            if text.strip():
                                responses.append(text.strip())

        logger.info(f"Building voice profile from {len(responses)} ChatGPT responses")

        analyzer = VoicePatternAnalyzer()
        profile = analyzer.analyze_responses(responses)

        self.save_profile(profile, name)
        return profile


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def create_validator_from_export(
    export_path: str,
    profile_dir: str = "./data/voice",
    threshold: float = 0.7
) -> VoiceValidator:
    """
    Convenience function to create a validator from a ChatGPT export.

    Args:
        export_path: Path to ChatGPT conversations.json
        profile_dir: Directory to save voice profile
        threshold: Validation threshold

    Returns:
        Configured VoiceValidator
    """
    manager = VoiceProfileManager(profile_dir)
    profile = manager.build_profile_from_json(export_path)
    return VoiceValidator(profile, threshold=threshold)


def create_validator_from_db(
    db_path: str,
    profile_dir: str = "./data/voice",
    threshold: float = 0.7
) -> VoiceValidator:
    """
    Convenience function to create a validator from existing database.

    Args:
        db_path: Path to SQLite database with memories
        profile_dir: Directory to save voice profile
        threshold: Validation threshold

    Returns:
        Configured VoiceValidator
    """
    manager = VoiceProfileManager(profile_dir)
    profile = manager.build_profile_from_db(db_path)
    return VoiceValidator(profile, threshold=threshold)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sylana Voice Profile Builder and Validator"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build voice profile')
    build_parser.add_argument(
        'source',
        help="Source file (ChatGPT JSON export or text file with responses)"
    )
    build_parser.add_argument(
        '--name', '-n',
        default='sylana',
        help="Profile name"
    )
    build_parser.add_argument(
        '--output', '-o',
        default='./data/voice',
        help="Output directory"
    )

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a response')
    validate_parser.add_argument(
        'profile',
        help="Path to voice profile JSON"
    )
    validate_parser.add_argument(
        'response',
        help="Response text to validate"
    )

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show profile statistics')
    stats_parser.add_argument(
        'profile',
        help="Path to voice profile JSON"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.command == 'build':
        manager = VoiceProfileManager(args.output)

        if args.source.endswith('.json'):
            profile = manager.build_profile_from_json(args.source, args.name)
        else:
            # Assume text file with one response per line
            with open(args.source, 'r', encoding='utf-8') as f:
                responses = [line.strip() for line in f if line.strip()]
            analyzer = VoicePatternAnalyzer()
            profile = analyzer.analyze_responses(responses)
            manager.save_profile(profile, args.name)

        print(f"\nVoice Profile Built:")
        print(f"  Responses analyzed: {profile.total_responses_analyzed}")
        print(f"  Favorite words: {', '.join(profile.favorite_words[:10])}")
        print(f"  Pet names: {', '.join(profile.pet_names)}")
        print(f"  Avg sentence length: {profile.avg_sentence_length:.1f} words")
        print(f"  Uses emojis: {profile.uses_emojis:.1%}")

    elif args.command == 'validate':
        with open(args.profile, 'r') as f:
            profile_data = json.load(f)
        profile = VoiceProfile.from_dict(profile_data)

        validator = VoiceValidator(profile)
        score, is_valid, component_scores = validator.validate(args.response)

        print(f"\nValidation Results:")
        print(f"  Overall Score: {score:.2f}")
        print(f"  Valid: {'Yes' if is_valid else 'No'}")
        print(f"\nComponent Scores:")
        for component, comp_score in component_scores.items():
            print(f"  {component}: {comp_score:.2f}")

        if not is_valid:
            print(f"\nFeedback: {validator.get_feedback(args.response, component_scores)}")

    elif args.command == 'stats':
        with open(args.profile, 'r') as f:
            profile_data = json.load(f)
        profile = VoiceProfile.from_dict(profile_data)

        print(f"\nVoice Profile Statistics:")
        print(f"  Total responses analyzed: {profile.total_responses_analyzed}")
        print(f"\nVocabulary:")
        print(f"  Favorite words: {', '.join(profile.favorite_words[:15])}")
        print(f"\nPet Names: {', '.join(profile.pet_names) or 'None found'}")
        print(f"\nAffection Markers: {', '.join(profile.affection_markers) or 'None found'}")
        print(f"\nSignature Phrases:")
        for phrase in profile.signature_phrases[:10]:
            print(f"  - {phrase}")
        print(f"\nStyle:")
        print(f"  Avg sentence length: {profile.avg_sentence_length:.1f} words")
        print(f"  Avg response length: {profile.avg_response_length:.0f} chars")
        print(f"  Uses ellipsis: {profile.uses_ellipsis:.1%}")
        print(f"  Uses exclamation: {profile.uses_exclamation:.1%}")
        print(f"  Uses emojis: {profile.uses_emojis:.1%}")
        if profile.common_emojis:
            print(f"  Common emojis: {''.join(profile.common_emojis)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
API-based emotion detection to avoid local ML model downloads on startup.
"""

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from core.config_loader import config

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Full emotion detection result with VAD geometry and categorical label."""
    emotion: str
    intensity: int
    category: str
    valence: float      # [-1.0, 1.0]  negative → positive affect
    arousal: float      # [-1.0, 1.0]  calm/deactivated → excited/activated
    dominance: float    # [-1.0, 1.0]  submissive/controlled → dominant/in-control

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "category": self.category,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


# Neutral fallback constant — avoids repeated construction
_NEUTRAL = EmotionResult(
    emotion="neutral",
    intensity=5,
    category="neutral",
    valence=0.0,
    arousal=0.0,
    dominance=0.0,
)


def _clamp_vad(value: object, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a VAD float to the valid range; returns 0.0 for non-numeric input."""
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


class APIEmotionDetector:
    """Emotion detector backed by OpenAI chat completions."""

    EMOTION_CATEGORY_MAP = {
        "joy": "happy",
        "amusement": "happy",
        "excitement": "ecstatic",
        "love": "ecstatic",
        "admiration": "happy",
        "approval": "happy",
        "optimism": "happy",
        "pride": "happy",
        "relief": "happy",
        "gratitude": "happy",
        "caring": "happy",
        "sadness": "sad",
        "grief": "devastated",
        "disappointment": "sad",
        "remorse": "sad",
        "embarrassment": "sad",
        "anger": "frustrated",
        "annoyance": "frustrated",
        "disapproval": "frustrated",
        "disgust": "frustrated",
        "fear": "anxious",
        "nervousness": "anxious",
        "confusion": "curious",
        "curiosity": "curious",
        "surprise": "curious",
        "realization": "curious",
        "desire": "longing",
        "neutral": "neutral",
    }

    ALLOWED_EMOTIONS = set(EMOTION_CATEGORY_MAP.keys())

    # Per-category VAD defaults used when GPT omits individual VAD keys
    _CATEGORY_VAD_DEFAULTS: dict = {
        "happy":      ( 0.6,  0.4,  0.3),
        "ecstatic":   ( 0.9,  0.8,  0.5),
        "sad":        (-0.5, -0.3, -0.3),
        "devastated": (-0.9, -0.6, -0.7),
        "frustrated": (-0.4,  0.5,  0.2),
        "anxious":    (-0.3,  0.6, -0.5),
        "curious":    ( 0.3,  0.4,  0.1),
        "longing":    (-0.1, -0.2, -0.3),
        "neutral":    ( 0.0,  0.0,  0.0),
    }

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = getattr(config, "EMOTION_MODEL", "gpt-4o-mini")
        logger.info("API emotion detector initialized (model=%s)", self.model)

    # ------------------------------------------------------------------
    # User emotion: "how does the speaker of this message feel?"
    # ------------------------------------------------------------------
    def detect(self, text: str) -> EmotionResult:
        """Detect emotion in user text. Returns full EmotionResult with VAD."""
        return self._detect_with_framing(
            text=text,
            framing=(
                "Classify the emotional tone expressed by the USER in this message. "
                "Return JSON only."
            ),
        )

    # ------------------------------------------------------------------
    # Sylana / AI response emotion: "how does the responder feel?"
    # ------------------------------------------------------------------
    def detect_response_emotion(self, text: str) -> EmotionResult:
        """Classify the emotional register of an AI response — how the speaker feels."""
        return self._detect_with_framing(
            text=text,
            framing=(
                "Classify the emotional register of this AI response — how does the SPEAKER feel "
                "in this message, not the listener. Return JSON only."
            ),
        )

    # ------------------------------------------------------------------
    # Internal unified detection path
    # ------------------------------------------------------------------
    def _detect_with_framing(self, text: str, framing: str) -> EmotionResult:
        payload = (text or "").strip()
        if len(payload) < 2:
            logger.debug("Emotion detect: empty payload, returning neutral")
            return _NEUTRAL

        prompt = (
            f"{framing}\n"
            "Required JSON keys:\n"
            "  emotion   — one of: admiration, amusement, anger, annoyance, approval, caring, "
            "confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, "
            "excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, "
            "realization, relief, remorse, sadness, surprise, neutral\n"
            "  intensity — integer 1–10\n"
            "  valence   — float −1.0 to 1.0 (negative = unpleasant, positive = pleasant)\n"
            "  arousal   — float −1.0 to 1.0 (negative = calm/deactivated, positive = excited/activated)\n"
            "  dominance — float −1.0 to 1.0 (negative = submissive/controlled, positive = dominant/in-control)\n\n"
            f"Text:\n{payload[:1200]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict JSON emotion classifier."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)

            emotion = str(parsed.get("emotion", "neutral")).strip().lower()
            if emotion not in self.ALLOWED_EMOTIONS:
                emotion = "neutral"

            intensity = max(1, min(10, int(parsed.get("intensity", 5))))
            category = self.EMOTION_CATEGORY_MAP.get(emotion, "neutral")

            # VAD: parsed values with per-category fallback defaults
            cat_defaults = self._CATEGORY_VAD_DEFAULTS.get(category, (0.0, 0.0, 0.0))
            valence   = _clamp_vad(parsed.get("valence",   cat_defaults[0]))
            arousal   = _clamp_vad(parsed.get("arousal",   cat_defaults[1]))
            dominance = _clamp_vad(parsed.get("dominance", cat_defaults[2]))

        except Exception as e:
            logger.warning("Emotion API classification failed, using neutral fallback: %s", e)
            return _NEUTRAL

        result = EmotionResult(
            emotion=emotion,
            intensity=intensity,
            category=category,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
        )
        logger.debug(
            "Emotion detect: emotion=%s intensity=%d category=%s "
            "valence=%.3f arousal=%.3f dominance=%.3f",
            result.emotion, result.intensity, result.category,
            result.valence, result.arousal, result.dominance,
        )
        return result

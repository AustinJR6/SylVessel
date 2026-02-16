"""
API-based emotion detection to avoid local ML model downloads on startup.
"""

import json
import logging
from typing import Tuple

from openai import OpenAI

from core.config_loader import config

logger = logging.getLogger(__name__)


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

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = getattr(config, "EMOTION_MODEL", "gpt-4o-mini")
        logger.info("API emotion detector initialized (model=%s)", self.model)

    def detect(self, text: str) -> Tuple[str, int, str]:
        """Return (emotion_label, intensity_1_to_10, category)."""
        payload = (text or "").strip()
        if len(payload) < 2:
            return "neutral", 5, "neutral"

        prompt = (
            "Classify the user's emotional tone.\n"
            "Return JSON only with keys: emotion, intensity.\n"
            "emotion must be one of: admiration, amusement, anger, annoyance, approval, caring, "
            "confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, "
            "excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, "
            "relief, remorse, sadness, surprise, neutral.\n"
            "intensity must be an integer 1-10.\n\n"
            f"Text:\n{payload[:1200]}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict JSON emotion classifier.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            emotion = str(parsed.get("emotion", "neutral")).strip().lower()
            intensity = int(parsed.get("intensity", 5))
        except Exception as e:
            logger.warning("Emotion API classification failed, using neutral fallback: %s", e)
            return "neutral", 5, "neutral"

        if emotion not in self.ALLOWED_EMOTIONS:
            emotion = "neutral"
        intensity = max(1, min(10, intensity))
        category = self.EMOTION_CATEGORY_MAP.get(emotion, "neutral")
        return emotion, intensity, category

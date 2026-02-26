from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Dict

from openai import OpenAI

from core.config_loader import config
from core.memory.memory_types import EmotionVector

logger = logging.getLogger(__name__)


class EmotionModel:
    """Continuous VAD scoring with backward-compatible category mapping."""

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = getattr(config, "EMOTION_MODEL", "gpt-4o-mini")

    @staticmethod
    def _category_from_vad(valence: float, arousal: float) -> str:
        if valence > 0.6 and arousal > 0.65:
            return "ecstatic"
        if valence > 0.2:
            return "happy"
        if valence < -0.6 and arousal > 0.65:
            return "devastated"
        if valence < -0.2:
            return "sad"
        return "neutral"

    def score(self, text: str) -> EmotionVector:
        payload = (text or "").strip()
        if not payload:
            return EmotionVector(valence=0.0, arousal=0.5, dominance=0.5, category="neutral", intensity=5)

        prompt = (
            "Return JSON with keys valence, arousal, dominance where valence in [-1,1] and arousal/dominance in [0,1]. "
            "Optional key intensity 1-10. No prose.\n\nText:\n" + payload[:1200]
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict emotion scorer."},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = json.loads(response.choices[0].message.content or "{}")
            valence = max(-1.0, min(1.0, float(parsed.get("valence", 0.0))))
            arousal = max(0.0, min(1.0, float(parsed.get("arousal", 0.5))))
            dominance = max(0.0, min(1.0, float(parsed.get("dominance", 0.5))))
            intensity = max(1, min(10, int(parsed.get("intensity", int((arousal * 9) + 1)))))
        except Exception as e:
            logger.warning("Emotion scoring failed, fallback neutral: %s", e)
            valence, arousal, dominance, intensity = 0.0, 0.5, 0.5, 5

        category = self._category_from_vad(valence, arousal)
        return EmotionVector(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            category=category,
            intensity=intensity,
        )

    def score_legacy(self, text: str) -> Dict[str, object]:
        vector = self.score(text)
        out = asdict(vector)
        out["emotion"] = vector.category
        return out

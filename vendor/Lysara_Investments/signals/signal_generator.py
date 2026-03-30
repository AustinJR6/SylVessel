from dataclasses import dataclass
from indicators.technical_indicators import relative_strength_index

@dataclass
class Signal:
    action: str
    confidence: float
    details: str

class SignalGenerator:
    """Combine technical and sentiment data into a trade signal."""

    def __init__(self, tech_weight: float = 0.7, sentiment_weight: float = 0.3):
        self.tech_weight = tech_weight
        self.sentiment_weight = sentiment_weight

    def generate(self, prices: list[float], sentiment_score: float) -> Signal:
        rsi = relative_strength_index(prices)
        action = "hold"
        base_conf = 0.0
        if rsi > 70:
            action = "sell"
            base_conf = (rsi - 50) / 50
        elif rsi < 30:
            action = "buy"
            base_conf = (50 - rsi) / 50

        final_conf = base_conf * self.tech_weight + sentiment_score * self.sentiment_weight
        final_conf = round(max(final_conf, 0.0), 3)
        details = f"rsi={rsi},sent={sentiment_score}"
        return Signal(action=action, confidence=final_conf, details=details)

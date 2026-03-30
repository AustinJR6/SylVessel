# utils/helpers.py

from datetime import datetime
import logging

def round_price(value: float, precision: int = 2) -> float:
    try:
        return round(float(value), precision)
    except Exception as e:
        logging.warning(f"Failed to round value: {e}")
        return 0.0

def format_timestamp(ts: datetime = None) -> str:
    """
    Returns ISO timestamp string. Defaults to now.
    """
    return (ts or datetime.utcnow()).isoformat()

def parse_price(data: dict, key: str = "price") -> float:
    """
    Extracts a price float safely from a data dict.
    """
    try:
        return float(data.get(key, 0))
    except (TypeError, ValueError):
        return 0.0

def safe_ratio(numerator: float, denominator: float) -> float:
    try:
        return numerator / denominator if denominator else 0.0
    except Exception:
        return 0.0

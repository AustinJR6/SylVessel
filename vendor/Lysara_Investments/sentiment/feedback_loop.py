"""Feedback loop for sentiment driven trades."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict

LOG_FILE = Path("logs/sentiment_performance.json")


def log_trade_result(ticker: str, success: bool) -> None:
    """Append trade outcome to log."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    data = []
    if LOG_FILE.exists():
        data = json.loads(LOG_FILE.read_text())
    data.append({"time": datetime.utcnow().isoformat(), "ticker": ticker, "success": success})
    LOG_FILE.write_text(json.dumps(data, indent=2))


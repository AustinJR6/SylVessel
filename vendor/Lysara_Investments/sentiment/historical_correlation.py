"""Correlation of sentiment spikes with historical price movements."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
from typing import List

import pandas as pd


ndefault_file = Path("data/sentiment_sensitivity.json")


def compute_correlation(sentiment_series: pd.Series, price_series: pd.Series) -> float:
    """Return correlation between sentiment and price change."""
    if len(sentiment_series) < 2 or len(price_series) < 2:
        return 0.0
    return float(sentiment_series.corr(price_series))


def save_sensitivity(ticker: str, value: float, file_path: Path = ndefault_file) -> None:
    """Persist the sentiment sensitivity for a ticker."""
    file_path.parent.mkdir(exist_ok=True)
    data = {}
    if file_path.exists():
        data = json.loads(file_path.read_text())
    data[ticker] = value
    file_path.write_text(json.dumps(data, indent=2))


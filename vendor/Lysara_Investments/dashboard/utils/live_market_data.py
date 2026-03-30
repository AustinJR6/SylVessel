"""Helpers to convert cached live prices into dashboard chart series."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import pandas as pd

from data.price_cache import get_price_series
from indicators.technical_indicators import moving_average, relative_strength_index


TIMEFRAME_TO_MINUTES = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def _to_df(series: list[dict]) -> pd.DataFrame:
    if not series:
        return pd.DataFrame(columns=["time", "price", "sma", "rsi"])
    df = pd.DataFrame(series).copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["time"]).sort_values("time")

    prices = df["price"].tolist()
    sma_window = min(20, max(2, len(prices)))
    sma_values = []
    rsi_values = []
    for i in range(len(prices)):
        cur = prices[: i + 1]
        sma_values.append(moving_average(cur, sma_window))
        rsi_values.append(relative_strength_index(cur, 14))
    df["sma"] = sma_values
    df["rsi"] = rsi_values
    return df


def get_live_chart_data(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
) -> list[dict]:
    """Return time-filtered live chart rows for Streamlit rendering."""

    series = get_price_series(symbol, limit=limit)
    df = _to_df(series)
    if df.empty:
        return []

    mins = TIMEFRAME_TO_MINUTES.get(timeframe, 60)
    cutoff = df["time"].max() - timedelta(minutes=mins)
    df = df[df["time"] >= cutoff]
    return df[["time", "price", "sma", "rsi"]].to_dict(orient="records")


def pick_symbol(candidates: Iterable[str], fallback: str) -> str:
    """Pick the first symbol that has cached data, else fallback."""
    for sym in candidates:
        if get_price_series(sym, limit=1):
            return sym
    return fallback

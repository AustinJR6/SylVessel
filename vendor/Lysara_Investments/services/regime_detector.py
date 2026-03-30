"""Market regime detection using ADX, ATR, and volatility percentiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeSignal:
    """Structured regime output."""

    label: str
    confidence: float
    adx: float
    atr: float
    volatility_percentile: float


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    tr = _true_range(df)
    return tr.rolling(period, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average Directional Index (Wilder-style approximation)."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = compute_atr(df, period=period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr.replace(0, np.nan)

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.rolling(period, min_periods=period).mean().fillna(0.0)


def detect_regime(
    price_history: pd.DataFrame,
    adx_period: int = 14,
    atr_period: int = 14,
    vol_window: int = 30,
) -> RegimeSignal:
    """Classify regime as trend/range/high_volatility with confidence.

    Required columns: ``high``, ``low``, ``close``.
    """

    required = {"high", "low", "close"}
    if not required.issubset(price_history.columns):
        raise ValueError(f"price_history must include {sorted(required)}")
    df = price_history.copy()
    if len(df) < max(adx_period * 2, vol_window + 5):
        return RegimeSignal("range", 0.3, 0.0, 0.0, 0.0)

    adx_series = compute_adx(df, period=adx_period)
    atr_series = compute_atr(df, period=atr_period)

    returns = df["close"].pct_change().fillna(0.0)
    rolling_vol = returns.rolling(vol_window, min_periods=vol_window).std()
    vol_now = float(rolling_vol.iloc[-1] or 0.0)
    vol_pct = float((rolling_vol <= vol_now).mean())

    adx_now = float(adx_series.iloc[-1] or 0.0)
    atr_now = float(atr_series.iloc[-1] or 0.0)

    if vol_pct >= 0.85:
        label = "high_volatility"
        confidence = min(1.0, 0.5 + (vol_pct - 0.85) * 2.0 + max(adx_now - 20, 0) / 100)
    elif adx_now >= 25:
        label = "trend"
        confidence = min(1.0, 0.45 + (adx_now - 25) / 40.0 + max(0.0, 0.6 - vol_pct) * 0.2)
    else:
        label = "range"
        confidence = min(1.0, 0.45 + (25 - adx_now) / 40.0 + max(0.0, 0.7 - vol_pct) * 0.2)

    return RegimeSignal(
        label=label,
        confidence=float(round(confidence, 4)),
        adx=float(round(adx_now, 4)),
        atr=float(round(atr_now, 6)),
        volatility_percentile=float(round(vol_pct, 4)),
    )


if __name__ == "__main__":
    idx = pd.date_range("2025-01-01", periods=120, freq="H")
    close = pd.Series(np.linspace(100, 120, len(idx)) + np.random.normal(0, 0.5, len(idx)), index=idx)
    frame = pd.DataFrame(
        {
            "high": close + np.random.uniform(0.1, 0.8, len(idx)),
            "low": close - np.random.uniform(0.1, 0.8, len(idx)),
            "close": close,
        }
    )
    print(detect_regime(frame))

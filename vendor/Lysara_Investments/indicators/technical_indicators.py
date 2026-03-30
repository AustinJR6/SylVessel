# indicators/technical_indicators.py

import pandas as pd

def moving_average(prices: list[float], period: int) -> float:
    """Simple moving average with basic safety checks."""
    if not prices:
        return 0.0
    if len(prices) < period:
        return sum(prices) / len(prices)
    return sum(prices[-period:]) / period

def relative_strength_index(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]

    avg_gain = sum(gains[-period:]) / period if gains else 0
    avg_loss = sum(losses[-period:]) / period if losses else 1e-6

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def exponential_moving_average(prices: list[float], period: int) -> float:
    df = pd.Series(prices)
    ema = df.ewm(span=period, adjust=False).mean()
    return float(round(ema.iloc[-1], 4))

def bollinger_bands(prices: list[float], period: int = 20, multiplier: float = 2.0):
    if len(prices) < period:
        return None, None

    ma = moving_average(prices, period)
    std_dev = pd.Series(prices[-period:]).std()
    upper = ma + (std_dev * multiplier)
    lower = ma - (std_dev * multiplier)
    return round(upper, 2), round(lower, 2)

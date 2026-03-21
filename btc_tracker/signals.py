"""
Technical analysis signals: MACD, RSI, Bollinger Bands, and trend detection.
Used to generate directional predictions for Kalshi BTC contracts.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MACDResult:
    macd_line: float
    signal_line: float
    histogram: float
    crossover: str  # "bullish", "bearish", or "none"


@dataclass
class SignalSummary:
    macd: Optional[MACDResult]
    rsi: Optional[float]
    rsi_signal: str  # "overbought", "oversold", "neutral"
    bb_position: Optional[float]  # 0-1 where in Bollinger Band
    bb_signal: str  # "upper", "lower", "middle"
    trend: str  # "up", "down", "sideways"
    trend_strength: float  # 0-1
    direction: str  # "higher" or "lower"
    confidence: float  # 0-1


def ema(prices, period):
    """Exponential Moving Average."""
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema_values = [np.mean(prices[:period])]
    for price in prices[period:]:
        ema_values.append((price - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def compute_macd(prices, fast=12, slow=26, signal_period=9):
    """Compute MACD indicator."""
    if len(prices) < slow + signal_period:
        return None

    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)

    if fast_ema is None or slow_ema is None:
        return None

    # Align lengths - fast_ema is longer, trim to match slow_ema
    offset = len(fast_ema) - len(slow_ema)
    fast_aligned = fast_ema[offset:]

    macd_line_values = [f - s for f, s in zip(fast_aligned, slow_ema)]

    if len(macd_line_values) < signal_period:
        return None

    signal_ema = ema(macd_line_values, signal_period)
    if signal_ema is None:
        return None

    macd_val = macd_line_values[-1]
    signal_val = signal_ema[-1]
    histogram = macd_val - signal_val

    # Detect crossover
    if len(macd_line_values) >= 2 and len(signal_ema) >= 2:
        prev_macd = macd_line_values[-2]
        prev_signal = signal_ema[-2]
        if prev_macd <= prev_signal and macd_val > signal_val:
            crossover = "bullish"
        elif prev_macd >= prev_signal and macd_val < signal_val:
            crossover = "bearish"
        else:
            crossover = "none"
    else:
        crossover = "none"

    return MACDResult(
        macd_line=round(macd_val, 2),
        signal_line=round(signal_val, 2),
        histogram=round(histogram, 2),
        crossover=crossover,
    )


def compute_rsi(prices, period=14):
    """Compute Relative Strength Index."""
    if len(prices) < period + 1:
        return None

    deltas = np.diff(prices[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def compute_bollinger(prices, period=20, num_std=2):
    """Compute Bollinger Band position (0=lower band, 1=upper band)."""
    if len(prices) < period:
        return None

    window = prices[-period:]
    mean = np.mean(window)
    std = np.std(window)

    if std == 0:
        return 0.5

    upper = mean + num_std * std
    lower = mean - num_std * std
    current = prices[-1]

    position = (current - lower) / (upper - lower)
    return round(np.clip(position, 0, 1), 3)


def detect_trend(prices, short_window=10, long_window=50):
    """Detect trend direction and strength using moving average crossover."""
    if len(prices) < long_window:
        if len(prices) < short_window:
            return "sideways", 0.0
        # Use what we have
        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices)
        diff_pct = (short_ma - long_ma) / long_ma * 100
    else:
        short_ma = np.mean(prices[-short_window:])
        long_ma = np.mean(prices[-long_window:])
        diff_pct = (short_ma - long_ma) / long_ma * 100

    strength = min(abs(diff_pct) / 2.0, 1.0)  # Normalize: 2% diff = full strength

    if diff_pct > 0.1:
        return "up", round(strength, 3)
    elif diff_pct < -0.1:
        return "down", round(strength, 3)
    else:
        return "sideways", 0.0


def generate_signals(prices):
    """Generate a complete signal summary from price history."""
    if len(prices) < 15:
        return SignalSummary(
            macd=None, rsi=None, rsi_signal="neutral",
            bb_position=None, bb_signal="middle",
            trend="sideways", trend_strength=0.0,
            direction="higher", confidence=0.0,
        )

    macd = compute_macd(prices)
    rsi = compute_rsi(prices)
    bb = compute_bollinger(prices)
    trend, trend_strength = detect_trend(prices)

    # RSI interpretation
    if rsi is not None:
        if rsi > 70:
            rsi_signal = "overbought"
        elif rsi < 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"
    else:
        rsi_signal = "neutral"

    # Bollinger interpretation
    if bb is not None:
        if bb > 0.8:
            bb_signal = "upper"
        elif bb < 0.2:
            bb_signal = "lower"
        else:
            bb_signal = "middle"
    else:
        bb_signal = "middle"

    # Combine signals into directional prediction with confidence
    votes_higher = 0.0
    votes_lower = 0.0
    total_weight = 0.0

    # MACD signal (weight: 3)
    if macd is not None:
        total_weight += 3
        if macd.histogram > 0:
            votes_higher += 3
        else:
            votes_lower += 3
        # Crossover bonus
        if macd.crossover == "bullish":
            votes_higher += 2
            total_weight += 2
        elif macd.crossover == "bearish":
            votes_lower += 2
            total_weight += 2

    # RSI signal (weight: 2)
    if rsi is not None:
        total_weight += 2
        if rsi_signal == "oversold":
            votes_higher += 2  # Mean reversion
        elif rsi_signal == "overbought":
            votes_lower += 2  # Mean reversion
        else:
            # Slight momentum following
            if rsi > 50:
                votes_higher += 1
            else:
                votes_lower += 1

    # Bollinger Band (weight: 2)
    if bb is not None:
        total_weight += 2
        if bb_signal == "lower":
            votes_higher += 2  # Bounce expected
        elif bb_signal == "upper":
            votes_lower += 2  # Pullback expected
        else:
            if bb > 0.5:
                votes_higher += 0.5
            else:
                votes_lower += 0.5

    # Trend (weight: 3)
    total_weight += 3
    if trend == "up":
        votes_higher += 3 * trend_strength
        votes_lower += 3 * (1 - trend_strength) * 0.3
    elif trend == "down":
        votes_lower += 3 * trend_strength
        votes_higher += 3 * (1 - trend_strength) * 0.3
    else:
        votes_higher += 1.5
        votes_lower += 1.5

    if total_weight == 0:
        direction = "higher"
        confidence = 0.0
    else:
        higher_pct = votes_higher / (votes_higher + votes_lower)
        if higher_pct >= 0.5:
            direction = "higher"
            confidence = round((higher_pct - 0.5) * 2, 3)  # Scale 0.5-1 -> 0-1
        else:
            direction = "lower"
            confidence = round((0.5 - higher_pct) * 2, 3)

    return SignalSummary(
        macd=macd,
        rsi=rsi,
        rsi_signal=rsi_signal,
        bb_position=bb,
        bb_signal=bb_signal,
        trend=trend,
        trend_strength=trend_strength,
        direction=direction,
        confidence=confidence,
    )

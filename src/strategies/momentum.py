"""
Momentum Expert Strategies
==========================

3 momentum strategies based on Phase 1 TA indicator survey.
These strategies capture time-series momentum and trend strength.
"""

import pandas as pd
import numpy as np

from .base import (
    ExpertStrategy,
    StrategyConfig,
    StrategyResult,
    Signal,
)


class MomentumStrategy(ExpertStrategy):
    """
    MO-01: 12-1 Momentum Strategy

    Entry: 12-month return > 0 (excluding most recent month)
    Exit: 12-month return < 0

    Regime Hypothesis: Time-series momentum - trends persist
    Failure Mode: Sharp reversals, mean-reversion periods

    Based on Moskowitz et al. (2012) "Time Series Momentum"
    """

    def __init__(self, lookback: int = 252, skip: int = 21):
        config = StrategyConfig(
            name="Momentum12-1",
            category="momentum",
            parameters={"lookback": lookback, "skip": skip},
            regime_hypothesis="Trending markets - momentum persistence",
            failure_mode="Sharp reversals, mean-reverting periods",
        )
        super().__init__(config)
        self.lookback = lookback  # ~12 months
        self.skip = skip  # ~1 month (avoid short-term reversal)

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.lookback:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        close = data["close"].iloc[:idx + 1]

        # 12-month return excluding last month
        start_price = close.iloc[-self.lookback]
        end_price = close.iloc[-self.skip] if self.skip > 0 else close.iloc[-1]

        momentum_return = (end_price / start_price) - 1

        if momentum_return > 0:
            signal = Signal.LONG
            # Higher momentum = higher confidence
            confidence = min(1.0, abs(momentum_return) * 2 + 0.5)
        else:
            signal = Signal.SHORT
            confidence = min(1.0, abs(momentum_return) * 2 + 0.5)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=momentum_return,
            metadata={
                "momentum_return": momentum_return,
                "start_price": start_price,
                "end_price": end_price,
            },
        )


class AroonTrendStrategy(ExpertStrategy):
    """
    MO-02: Aroon Trend Strategy

    Entry: Aroon Up > 70 and Aroon Down < 30 (strong uptrend)
    Exit: Opposite conditions or crossover

    Regime Hypothesis: Trend initiation and continuation
    Failure Mode: Choppy, trendless markets
    """

    def __init__(self, period: int = 25):
        config = StrategyConfig(
            name="AroonTrend",
            category="momentum",
            parameters={"period": period},
            regime_hypothesis="Trend identification and initiation",
            failure_mode="Choppy markets with no clear trend",
        )
        super().__init__(config)
        self.period = period

    def _calculate_aroon(self, high: pd.Series, low: pd.Series) -> tuple:
        """Calculate Aroon Up and Aroon Down."""
        # Days since highest high
        rolling_high = high.rolling(self.period + 1)
        days_since_high = self.period - rolling_high.apply(
            lambda x: x.argmax() if len(x) > 0 else 0
        )
        aroon_up = 100 * (self.period - days_since_high) / self.period

        # Days since lowest low
        rolling_low = low.rolling(self.period + 1)
        days_since_low = self.period - rolling_low.apply(
            lambda x: x.argmin() if len(x) > 0 else 0
        )
        aroon_down = 100 * (self.period - days_since_low) / self.period

        return aroon_up.iloc[-1], aroon_down.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period + 1:
            return StrategyResult(Signal.CASH, 0.0, 50.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]

        aroon_up, aroon_down = self._calculate_aroon(high, low)

        if pd.isna(aroon_up) or pd.isna(aroon_down):
            return StrategyResult(Signal.CASH, 0.0, 50.0, {"reason": "insufficient_data"})

        aroon_oscillator = aroon_up - aroon_down

        if aroon_up > 70 and aroon_down < 30:
            signal = Signal.LONG
            confidence = min(1.0, aroon_up / 100)
        elif aroon_down > 70 and aroon_up < 30:
            signal = Signal.SHORT
            confidence = min(1.0, aroon_down / 100)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=aroon_oscillator,
            metadata={
                "aroon_up": aroon_up,
                "aroon_down": aroon_down,
                "aroon_oscillator": aroon_oscillator,
            },
        )


class TRIXTrendStrategy(ExpertStrategy):
    """
    MO-03: TRIX Trend Strategy

    Entry: TRIX > Signal line
    Exit: TRIX < Signal line

    Regime Hypothesis: Smoothed momentum for trend changes
    Failure Mode: Fast, choppy markets (TRIX is lagging)

    TRIX is triple-smoothed EMA - good for filtering noise
    """

    def __init__(self, period: int = 15, signal_period: int = 9):
        config = StrategyConfig(
            name="TRIXTrend",
            category="momentum",
            parameters={"period": period, "signal_period": signal_period},
            regime_hypothesis="Filtered trend identification",
            failure_mode="Fast markets - TRIX is lagging",
        )
        super().__init__(config)
        self.period = period
        self.signal_period = signal_period

    def _calculate_trix(self, close: pd.Series) -> tuple:
        """Calculate TRIX and signal line."""
        # Triple EMA
        ema1 = close.ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()

        # TRIX = percentage change of triple EMA
        trix = ema3.pct_change() * 100

        # Signal line = EMA of TRIX
        signal = trix.ewm(span=self.signal_period, adjust=False).mean()

        return trix.iloc[-1], signal.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        min_periods = self.period * 3 + self.signal_period
        if idx < min_periods:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        close = data["close"].iloc[:idx + 1]

        trix, signal_line = self._calculate_trix(close)

        if pd.isna(trix) or pd.isna(signal_line):
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "insufficient_data"})

        histogram = trix - signal_line

        if trix > signal_line:
            signal = Signal.LONG
            confidence = min(1.0, abs(histogram) * 10 + 0.5)
        else:
            signal = Signal.SHORT
            confidence = min(1.0, abs(histogram) * 10 + 0.5)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=trix,
            metadata={
                "trix": trix,
                "signal_line": signal_line,
                "histogram": histogram,
            },
        )

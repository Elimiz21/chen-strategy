"""
Mean-Reversion Expert Strategies
================================

5 mean-reversion strategies based on Phase 1 TA indicator survey.
These strategies profit from price returning to mean after overextension.
"""

import pandas as pd
import numpy as np

from .base import (
    ExpertStrategy,
    StrategyConfig,
    StrategyResult,
    Signal,
)


class RSIReversalStrategy(ExpertStrategy):
    """
    MR-01: RSI Reversal Strategy

    Entry: RSI < 30 (oversold)
    Exit: RSI > 50 or RSI > 70 (overbought)

    Regime Hypothesis: Works in oversold bounces, ranging markets
    Failure Mode: Strong trends where RSI stays oversold/overbought
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        config = StrategyConfig(
            name="RSIReversal",
            category="mean_reversion",
            parameters={"period": period, "oversold": oversold, "overbought": overbought},
            regime_hypothesis="Oversold bounces in ranging/volatile markets",
            failure_mode="Strong trends where oversold conditions persist",
        )
        super().__init__(config)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _calculate_rsi(self, close: pd.Series) -> float:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period + 1:
            return StrategyResult(Signal.CASH, 0.0, 50.0, {"reason": "warmup"})

        close = data["close"].iloc[:idx + 1]
        rsi = self._calculate_rsi(close)

        if pd.isna(rsi):
            return StrategyResult(Signal.CASH, 0.0, 50.0, {"reason": "insufficient_data"})

        if rsi < self.oversold:
            signal = Signal.LONG
            # Higher confidence when more oversold
            confidence = min(1.0, (self.oversold - rsi) / self.oversold + 0.5)
        elif rsi > self.overbought:
            signal = Signal.SHORT
            confidence = min(1.0, (rsi - self.overbought) / (100 - self.overbought) + 0.5)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=rsi,
            metadata={"rsi": rsi},
        )


class BollingerBounceStrategy(ExpertStrategy):
    """
    MR-02: Bollinger Band Bounce Strategy

    Entry: Price touches lower band
    Exit: Price returns to middle band or touches upper band

    Regime Hypothesis: Low-vol consolidation, ranging markets
    Failure Mode: Trending markets where price "walks the band"
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        config = StrategyConfig(
            name="BollingerBounce",
            category="mean_reversion",
            parameters={"period": period, "std_dev": std_dev},
            regime_hypothesis="Low-volatility consolidation phases",
            failure_mode="Strong trends - price walks along band",
        )
        super().__init__(config)
        self.period = period
        self.std_dev = std_dev

    def _calculate_bands(self, close: pd.Series) -> tuple:
        """Calculate Bollinger Bands."""
        middle = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        return middle.iloc[-1], upper.iloc[-1], lower.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period:
            return StrategyResult(Signal.CASH, 0.0, 0.5, {"reason": "warmup"})

        close = data["close"].iloc[:idx + 1]
        current_price = close.iloc[-1]

        middle, upper, lower = self._calculate_bands(close)

        if pd.isna(middle):
            return StrategyResult(Signal.CASH, 0.0, 0.5, {"reason": "insufficient_data"})

        # Calculate %B (position within bands)
        bandwidth = upper - lower
        percent_b = (current_price - lower) / bandwidth if bandwidth > 0 else 0.5

        if current_price <= lower:
            signal = Signal.LONG
            confidence = min(1.0, (lower - current_price) / lower * 20 + 0.6)
        elif current_price >= upper:
            signal = Signal.SHORT
            confidence = min(1.0, (current_price - upper) / upper * 20 + 0.6)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=percent_b,
            metadata={
                "middle": middle,
                "upper": upper,
                "lower": lower,
                "percent_b": percent_b,
            },
        )


class StochasticStrategy(ExpertStrategy):
    """
    MR-03: Stochastic Oscillator Strategy

    Entry: K < 20 and K crosses above D
    Exit: K > 80

    Regime Hypothesis: Ranging, oscillating markets
    Failure Mode: Strong trends where stochastic stays extreme
    """

    def __init__(self, k_period: int = 14, d_period: int = 3, smooth: int = 3):
        config = StrategyConfig(
            name="Stochastic",
            category="mean_reversion",
            parameters={"k_period": k_period, "d_period": d_period, "smooth": smooth},
            regime_hypothesis="Ranging markets with clear oscillations",
            failure_mode="Trending markets - stochastic stays pinned",
        )
        super().__init__(config)
        self.k_period = k_period
        self.d_period = d_period
        self.smooth = smooth

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
        """Calculate Stochastic %K and %D."""
        lowest_low = low.rolling(self.k_period).min()
        highest_high = high.rolling(self.k_period).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = raw_k.rolling(self.smooth).mean()  # Smoothed %K
        d = k.rolling(self.d_period).mean()  # %D (signal)

        return k.iloc[-1], d.iloc[-1], k.iloc[-2] if len(k) > 1 else k.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.k_period + self.d_period + self.smooth:
            return StrategyResult(Signal.CASH, 0.0, 50.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        k, d, k_prev = self._calculate_stochastic(high, low, close)

        if pd.isna(k) or pd.isna(d):
            return StrategyResult(Signal.CASH, 0.0, 50.0, {"reason": "insufficient_data"})

        # Oversold with bullish crossover
        if k < 20 and k > k_prev:  # K rising from oversold
            signal = Signal.LONG
            confidence = min(1.0, (20 - k) / 20 + 0.5)
        # Overbought with bearish crossover
        elif k > 80 and k < k_prev:  # K falling from overbought
            signal = Signal.SHORT
            confidence = min(1.0, (k - 80) / 20 + 0.5)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=k,
            metadata={"k": k, "d": d},
        )


class WilliamsRStrategy(ExpertStrategy):
    """
    MR-04: Williams %R Strategy

    Entry: %R < -80 (oversold)
    Exit: %R > -20 (overbought)

    Regime Hypothesis: Oscillating, ranging markets
    Failure Mode: Trending markets with persistent extremes
    """

    def __init__(self, period: int = 14):
        config = StrategyConfig(
            name="WilliamsR",
            category="mean_reversion",
            parameters={"period": period},
            regime_hypothesis="Ranging markets with price oscillations",
            failure_mode="Strong trends - indicator stays extreme",
        )
        super().__init__(config)
        self.period = period

    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate Williams %R."""
        highest_high = high.iloc[-self.period:].max()
        lowest_low = low.iloc[-self.period:].min()

        williams_r = -100 * (highest_high - close.iloc[-1]) / (highest_high - lowest_low)
        return williams_r

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period:
            return StrategyResult(Signal.CASH, 0.0, -50.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        williams_r = self._calculate_williams_r(high, low, close)

        if pd.isna(williams_r):
            return StrategyResult(Signal.CASH, 0.0, -50.0, {"reason": "insufficient_data"})

        if williams_r < -80:
            signal = Signal.LONG
            confidence = min(1.0, (-80 - williams_r) / 20 + 0.6)
        elif williams_r > -20:
            signal = Signal.SHORT
            confidence = min(1.0, (williams_r + 20) / 20 + 0.6)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=williams_r,
            metadata={"williams_r": williams_r},
        )


class CCIReversalStrategy(ExpertStrategy):
    """
    MR-05: Commodity Channel Index Reversal Strategy

    Entry: CCI < -100 (oversold)
    Exit: CCI > 0 (neutral) or CCI > 100 (overbought)

    Regime Hypothesis: Mean-reverting markets after extreme moves
    Failure Mode: Strong trends with sustained CCI extremes
    """

    def __init__(self, period: int = 20):
        config = StrategyConfig(
            name="CCIReversal",
            category="mean_reversion",
            parameters={"period": period},
            regime_hypothesis="Mean-reverting after extreme deviations",
            failure_mode="Trending markets with sustained extremes",
        )
        super().__init__(config)
        self.period = period

    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(self.period).mean()
        mean_deviation = typical_price.rolling(self.period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )

        cci = (typical_price.iloc[-1] - sma.iloc[-1]) / (0.015 * mean_deviation.iloc[-1])
        return cci

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        cci = self._calculate_cci(high, low, close)

        if pd.isna(cci):
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "insufficient_data"})

        if cci < -100:
            signal = Signal.LONG
            confidence = min(1.0, (-100 - cci) / 100 + 0.5)
        elif cci > 100:
            signal = Signal.SHORT
            confidence = min(1.0, (cci - 100) / 100 + 0.5)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=cci,
            metadata={"cci": cci},
        )

"""
Volume-Based Expert Strategies
==============================

3 volume-based strategies based on Phase 1 TA indicator survey.
These strategies use volume dynamics to confirm price moves.
"""

import pandas as pd
import numpy as np

from .base import (
    ExpertStrategy,
    StrategyConfig,
    StrategyResult,
    Signal,
)


class OBVConfirmationStrategy(ExpertStrategy):
    """
    VO-01: On-Balance Volume Confirmation Strategy

    Entry: Price rising AND OBV rising (confirmed uptrend)
    Exit: Divergence between price and OBV

    Regime Hypothesis: Volume-confirmed trends
    Failure Mode: Low volume environments, divergence fakeouts
    """

    def __init__(self, lookback: int = 20):
        config = StrategyConfig(
            name="OBVConfirmation",
            category="volume",
            parameters={"lookback": lookback},
            regime_hypothesis="Volume-confirmed trending markets",
            failure_mode="Low volume periods, false divergences",
        )
        super().__init__(config)
        self.lookback = lookback

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        direction = np.sign(close.diff())
        obv = (direction * volume).cumsum()
        return obv

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.lookback:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        close = data["close"].iloc[:idx + 1]
        volume = data["volume"].iloc[:idx + 1]

        obv = self._calculate_obv(close, volume)

        # Calculate trends over lookback period
        price_change = close.iloc[-1] / close.iloc[-self.lookback] - 1
        obv_change = obv.iloc[-1] - obv.iloc[-self.lookback]
        obv_change_normalized = obv_change / volume.iloc[-self.lookback:].mean()

        # Determine signal based on price/OBV alignment
        price_up = price_change > 0.01  # 1% threshold
        price_down = price_change < -0.01
        obv_up = obv_change_normalized > 0
        obv_down = obv_change_normalized < 0

        if price_up and obv_up:
            signal = Signal.LONG
            confidence = min(1.0, abs(price_change) * 10 + 0.5)
        elif price_down and obv_down:
            signal = Signal.SHORT
            confidence = min(1.0, abs(price_change) * 10 + 0.5)
        elif price_up and obv_down:
            # Bearish divergence - price up but OBV down
            signal = Signal.CASH
            confidence = 0.4
        elif price_down and obv_up:
            # Bullish divergence - price down but OBV up
            signal = Signal.CASH
            confidence = 0.4
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=obv.iloc[-1],
            metadata={
                "obv": obv.iloc[-1],
                "price_change": price_change,
                "obv_change_normalized": obv_change_normalized,
            },
        )


class MFIReversalStrategy(ExpertStrategy):
    """
    VO-02: Money Flow Index Reversal Strategy

    Entry: MFI < 20 (oversold with volume)
    Exit: MFI > 80 (overbought)

    Regime Hypothesis: High-volume reversals
    Failure Mode: Low volume periods where MFI is unreliable
    """

    def __init__(self, period: int = 14):
        config = StrategyConfig(
            name="MFIReversal",
            category="volume",
            parameters={"period": period},
            regime_hypothesis="High-volume reversal points",
            failure_mode="Low volume periods, trending markets",
        )
        super().__init__(config)
        self.period = period

    def _calculate_mfi(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> float:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        # Direction of flow
        tp_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_diff > 0, 0)
        negative_flow = raw_money_flow.where(tp_diff < 0, 0)

        positive_mf = positive_flow.rolling(self.period).sum()
        negative_mf = negative_flow.rolling(self.period).sum()

        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period + 1:
            return StrategyResult(Signal.CASH, 0.0, 50.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]
        volume = data["volume"].iloc[:idx + 1]

        mfi = self._calculate_mfi(high, low, close, volume)

        if pd.isna(mfi):
            return StrategyResult(Signal.CASH, 0.0, 50.0, {"reason": "insufficient_data"})

        if mfi < 20:
            signal = Signal.LONG
            confidence = min(1.0, (20 - mfi) / 20 + 0.6)
        elif mfi > 80:
            signal = Signal.SHORT
            confidence = min(1.0, (mfi - 80) / 20 + 0.6)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=mfi,
            metadata={"mfi": mfi},
        )


class VWAPReversionStrategy(ExpertStrategy):
    """
    VO-03: VWAP Reversion Strategy

    Entry: Price < VWAP - 2 std (long) or Price > VWAP + 2 std (short)
    Exit: Price returns to VWAP

    Regime Hypothesis: Intraday/daily mean-reversion around VWAP
    Failure Mode: Strong trending days where price stays extended
    """

    def __init__(self, period: int = 20, std_mult: float = 2.0):
        config = StrategyConfig(
            name="VWAPReversion",
            category="volume",
            parameters={"period": period, "std_mult": std_mult},
            regime_hypothesis="Mean-reversion around volume-weighted price",
            failure_mode="Strong trends, price walks away from VWAP",
        )
        super().__init__(config)
        self.period = period
        self.std_mult = std_mult

    def _calculate_vwap(
        self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> tuple:
        """Calculate rolling VWAP and standard deviation bands."""
        typical_price = (high + low + close) / 3

        # Rolling VWAP
        cum_vol = volume.rolling(self.period).sum()
        cum_tp_vol = (typical_price * volume).rolling(self.period).sum()
        vwap = cum_tp_vol / cum_vol

        # Standard deviation of price from VWAP
        deviation = typical_price - vwap
        std = deviation.rolling(self.period).std()

        return vwap.iloc[-1], std.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]
        volume = data["volume"].iloc[:idx + 1]

        current_price = close.iloc[-1]
        vwap, std = self._calculate_vwap(high, low, close, volume)

        if pd.isna(vwap) or pd.isna(std) or std == 0:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "insufficient_data"})

        upper_band = vwap + (self.std_mult * std)
        lower_band = vwap - (self.std_mult * std)

        # Z-score from VWAP
        z_score = (current_price - vwap) / std if std > 0 else 0

        if current_price < lower_band:
            signal = Signal.LONG
            confidence = min(1.0, abs(z_score) / 4 + 0.5)
        elif current_price > upper_band:
            signal = Signal.SHORT
            confidence = min(1.0, abs(z_score) / 4 + 0.5)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=z_score,
            metadata={
                "vwap": vwap,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "z_score": z_score,
            },
        )

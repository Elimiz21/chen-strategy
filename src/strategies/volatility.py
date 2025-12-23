"""
Volatility-Based Expert Strategies
==================================

4 volatility-based strategies based on Phase 1 TA indicator survey.
These strategies use volatility dynamics for entry/exit and position sizing.
"""

import pandas as pd
import numpy as np

from .base import (
    ExpertStrategy,
    StrategyConfig,
    StrategyResult,
    Signal,
)


class ATRBreakoutStrategy(ExpertStrategy):
    """
    VB-01: ATR Breakout Strategy

    Entry: Price breaks above upper ATR band (long) or below lower band (short)
    Exit: Price returns inside bands

    Regime Hypothesis: Volatility breakouts, trending environments
    Failure Mode: False breakouts in low-volatility consolidation
    """

    def __init__(self, period: int = 14, multiplier: float = 2.0):
        config = StrategyConfig(
            name="ATRBreakout",
            category="volatility",
            parameters={"period": period, "multiplier": multiplier},
            regime_hypothesis="Volatility breakouts in trending markets",
            failure_mode="False breakouts, choppy consolidation",
        )
        super().__init__(config)
        self.period = period
        self.multiplier = multiplier

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.period).mean()
        return atr.iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period + 1:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        current_price = close.iloc[-1]
        atr = self._calculate_atr(high, low, close)

        if pd.isna(atr):
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "insufficient_data"})

        # Calculate bands around moving average
        sma = close.iloc[-self.period:].mean()
        upper_band = sma + (self.multiplier * atr)
        lower_band = sma - (self.multiplier * atr)

        if current_price > upper_band:
            signal = Signal.LONG
            distance = (current_price - upper_band) / atr
            confidence = min(1.0, distance * 0.3 + 0.6)
        elif current_price < lower_band:
            signal = Signal.SHORT
            distance = (lower_band - current_price) / atr
            confidence = min(1.0, distance * 0.3 + 0.6)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=atr,
            metadata={
                "atr": atr,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "sma": sma,
            },
        )


class KeltnerBreakoutStrategy(ExpertStrategy):
    """
    VB-02: Keltner Channel Breakout Strategy

    Entry: Price breaks above upper Keltner channel
    Exit: Price returns below EMA

    Regime Hypothesis: Trending breakouts with volatility confirmation
    Failure Mode: Ranging markets with false breakouts
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        config = StrategyConfig(
            name="KeltnerBreakout",
            category="volatility",
            parameters={
                "ema_period": ema_period,
                "atr_period": atr_period,
                "multiplier": multiplier,
            },
            regime_hypothesis="Trending breakouts with vol expansion",
            failure_mode="False breakouts in ranging markets",
        )
        super().__init__(config)
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> float:
        """Calculate ATR."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean().iloc[-1]

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < max(self.ema_period, self.atr_period) + 1:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        current_price = close.iloc[-1]

        # EMA (middle line)
        ema = close.ewm(span=self.ema_period, adjust=False).mean().iloc[-1]

        # ATR for bands
        atr = self._calculate_atr(high, low, close)

        if pd.isna(ema) or pd.isna(atr):
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "insufficient_data"})

        upper = ema + (self.multiplier * atr)
        lower = ema - (self.multiplier * atr)

        if current_price > upper:
            signal = Signal.LONG
            confidence = min(1.0, (current_price - upper) / atr * 0.5 + 0.6)
        elif current_price < lower:
            signal = Signal.SHORT
            confidence = min(1.0, (lower - current_price) / atr * 0.5 + 0.6)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=atr,
            metadata={"ema": ema, "upper": upper, "lower": lower, "atr": atr},
        )


class VolTargetingStrategy(ExpertStrategy):
    """
    VB-03: Volatility Targeting Strategy

    Position Size = Target Vol / Realized Vol
    Always long with vol-adjusted sizing

    Regime Hypothesis: Works across all regimes via position sizing
    Failure Mode: Rapid vol regime changes (vol spike after sizing)
    """

    def __init__(self, target_vol: float = 0.15, vol_lookback: int = 20, max_leverage: float = 3.0):
        config = StrategyConfig(
            name="VolTargeting",
            category="volatility",
            parameters={
                "target_vol": target_vol,
                "vol_lookback": vol_lookback,
                "max_leverage": max_leverage,
            },
            regime_hypothesis="All regimes - risk-adjusted exposure",
            failure_mode="Rapid vol spikes after position sizing",
        )
        super().__init__(config)
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.vol_lookback:
            return StrategyResult(Signal.LONG, 0.5, 0.0, {"reason": "warmup"})

        close = data["close"].iloc[:idx + 1]

        # Calculate realized volatility (annualized)
        returns = close.pct_change().iloc[-self.vol_lookback:]
        realized_vol = returns.std() * np.sqrt(252)

        if pd.isna(realized_vol) or realized_vol == 0:
            return StrategyResult(Signal.LONG, 0.5, 0.0, {"reason": "insufficient_data"})

        # Vol-adjusted leverage
        vol_multiplier = self.target_vol / realized_vol
        vol_multiplier = min(vol_multiplier, self.max_leverage)
        vol_multiplier = max(vol_multiplier, 0.1)  # Minimum 10% position

        # Confidence scales with how "normal" volatility is
        vol_ratio = realized_vol / self.target_vol
        if 0.5 < vol_ratio < 2.0:
            confidence = 0.8
        elif 0.25 < vol_ratio < 4.0:
            confidence = 0.6
        else:
            confidence = 0.4

        return StrategyResult(
            signal=Signal.LONG,
            confidence=confidence,
            raw_indicator=realized_vol,
            metadata={
                "realized_vol": realized_vol,
                "vol_multiplier": vol_multiplier,
                "target_vol": self.target_vol,
            },
        )

    def get_position_size(
        self,
        signal: Signal,
        confidence: float,
        max_leverage: float = 3.0,
    ) -> float:
        """Override to use vol-adjusted sizing."""
        # This will be called with the result from generate_signal
        # The actual vol-adjusted size should come from metadata
        if signal == Signal.CASH:
            return 0.0
        return 1.0  # Base size, actual size from metadata.vol_multiplier


class BBSqueezeStrategy(ExpertStrategy):
    """
    VB-04: Bollinger Band Squeeze Strategy

    Entry: BB squeeze (narrow bands) followed by breakout
    Exit: Opposite direction break

    Regime Hypothesis: Pre-breakout consolidation periods
    Failure Mode: False breakouts, squeeze without follow-through
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
    ):
        config = StrategyConfig(
            name="BBSqueeze",
            category="volatility",
            parameters={
                "bb_period": bb_period,
                "bb_std": bb_std,
                "kc_period": kc_period,
                "kc_mult": kc_mult,
            },
            regime_hypothesis="Pre-breakout consolidation phases",
            failure_mode="False breakouts, no follow-through",
        )
        super().__init__(config)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult

    def _calculate_bands(
        self, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> dict:
        """Calculate BB and KC bands."""
        # Bollinger Bands
        bb_middle = close.rolling(self.bb_period).mean()
        bb_std = close.rolling(self.bb_period).std()
        bb_upper = bb_middle + (bb_std * self.bb_std)
        bb_lower = bb_middle - (bb_std * self.bb_std)

        # Keltner Channels
        kc_middle = close.ewm(span=self.kc_period, adjust=False).mean()
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.kc_period).mean()
        kc_upper = kc_middle + (atr * self.kc_mult)
        kc_lower = kc_middle - (atr * self.kc_mult)

        return {
            "bb_upper": bb_upper.iloc[-1],
            "bb_lower": bb_lower.iloc[-1],
            "kc_upper": kc_upper.iloc[-1],
            "kc_lower": kc_lower.iloc[-1],
            "bb_middle": bb_middle.iloc[-1],
        }

    def _calculate_momentum(self, close: pd.Series, period: int = 12) -> float:
        """Calculate momentum oscillator."""
        highest = close.rolling(period).max()
        lowest = close.rolling(period).min()
        momentum = (close - (highest + lowest) / 2).iloc[-1]
        return momentum

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        lookback = max(self.bb_period, self.kc_period) + 1
        if idx < lookback:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        bands = self._calculate_bands(high, low, close)

        if any(pd.isna(v) for v in bands.values()):
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "insufficient_data"})

        # Squeeze detection: BB inside KC
        squeeze_on = (bands["bb_lower"] > bands["kc_lower"]) and (
            bands["bb_upper"] < bands["kc_upper"]
        )

        # Momentum for direction
        momentum = self._calculate_momentum(close)

        if squeeze_on:
            # In squeeze - wait for breakout direction
            if momentum > 0:
                signal = Signal.LONG
                confidence = 0.6
            else:
                signal = Signal.SHORT
                confidence = 0.6
        else:
            # Out of squeeze - momentum determines position
            if momentum > 0:
                signal = Signal.LONG
                confidence = 0.5
            elif momentum < 0:
                signal = Signal.SHORT
                confidence = 0.5
            else:
                signal = Signal.CASH
                confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=float(squeeze_on),
            metadata={
                "squeeze_on": squeeze_on,
                "momentum": momentum,
                **bands,
            },
        )

"""
Trend-Following Expert Strategies
=================================

6 trend-following strategies based on Phase 1 TA indicator survey.
Each strategy has a clear regime hypothesis for when it should work.
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base import (
    ExpertStrategy,
    StrategyConfig,
    StrategyResult,
    Signal,
)


class GoldenCrossStrategy(ExpertStrategy):
    """
    TF-01: Golden Cross Strategy

    Entry: 50 SMA > 200 SMA
    Exit: 50 SMA < 200 SMA

    Regime Hypothesis: Works best in sustained bull trends (Bull-Calm, Bull-Normal)
    Failure Mode: Whipsaws in sideways/transition markets
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        config = StrategyConfig(
            name="GoldenCross",
            category="trend_following",
            parameters={"fast_period": fast_period, "slow_period": slow_period},
            regime_hypothesis="Bull trends - sustained upward momentum",
            failure_mode="Sideways markets with repeated SMA crosses",
        )
        super().__init__(config)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.slow_period:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        # Calculate SMAs using only data up to idx (no look-ahead)
        close = data["close"].iloc[:idx + 1]
        sma_fast = close.iloc[-self.fast_period:].mean()
        sma_slow = close.iloc[-self.slow_period:].mean()

        ratio = sma_fast / sma_slow

        if sma_fast > sma_slow:
            signal = Signal.LONG
            # Confidence based on strength of crossover
            confidence = min(1.0, (ratio - 1) * 20 + 0.5)
        else:
            signal = Signal.CASH
            confidence = min(1.0, (1 - ratio) * 20 + 0.5)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=ratio,
            metadata={"sma_fast": sma_fast, "sma_slow": sma_slow},
        )


class MACDTrendStrategy(ExpertStrategy):
    """
    TF-02: MACD Trend Strategy

    Entry: MACD > Signal Line
    Exit: MACD < Signal Line

    Regime Hypothesis: Works in trending markets (Bull or Bear)
    Failure Mode: Choppy, range-bound markets
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        config = StrategyConfig(
            name="MACDTrend",
            category="trend_following",
            parameters={"fast": fast, "slow": slow, "signal": signal},
            regime_hypothesis="Trending markets - momentum confirmation",
            failure_mode="Range-bound with false crossovers",
        )
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA without look-ahead."""
        return series.ewm(span=period, adjust=False).mean()

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.slow + self.signal_period:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        # Use only data up to idx
        close = data["close"].iloc[:idx + 1]

        # Calculate MACD components
        ema_fast = self._ema(close, self.fast).iloc[-1]
        ema_slow = self._ema(close, self.slow).iloc[-1]
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        macd_series = self._ema(close, self.fast) - self._ema(close, self.slow)
        signal_line = self._ema(macd_series, self.signal_period).iloc[-1]

        histogram = macd_line - signal_line

        if macd_line > signal_line:
            signal = Signal.LONG
            confidence = min(1.0, abs(histogram) / close.iloc[-1] * 100 + 0.5)
        elif macd_line < signal_line:
            # Can go short in bear trends
            signal = Signal.SHORT
            confidence = min(1.0, abs(histogram) / close.iloc[-1] * 100 + 0.5)
        else:
            signal = Signal.CASH
            confidence = 0.5

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=histogram,
            metadata={"macd": macd_line, "signal": signal_line, "histogram": histogram},
        )


class ADXBreakoutStrategy(ExpertStrategy):
    """
    TF-03: ADX Breakout Strategy

    Entry: ADX > 25 and +DI > -DI (long) or -DI > +DI (short)
    Exit: ADX < 20 or directional change

    Regime Hypothesis: Strong trending markets only
    Failure Mode: Low ADX (non-trending) environments
    """

    def __init__(self, period: int = 14, adx_threshold: int = 25):
        config = StrategyConfig(
            name="ADXBreakout",
            category="trend_following",
            parameters={"period": period, "adx_threshold": adx_threshold},
            regime_hypothesis="Strong trends with ADX > 25",
            failure_mode="Weak trends (ADX < 20), ranging markets",
        )
        super().__init__(config)
        self.period = period
        self.adx_threshold = adx_threshold

    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
        """Calculate ADX, +DI, -DI."""
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.period).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Smoothed DI
        plus_di = 100 * (plus_dm.rolling(self.period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.period).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(self.period).mean()

        return adx, plus_di, minus_di

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.period * 3:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        # Use only data up to idx
        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        adx, plus_di, minus_di = self._calculate_adx(high, low, close)

        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]

        if pd.isna(current_adx):
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "insufficient_data"})

        if current_adx > self.adx_threshold:
            if current_plus_di > current_minus_di:
                signal = Signal.LONG
                confidence = min(1.0, current_adx / 50)
            else:
                signal = Signal.SHORT
                confidence = min(1.0, current_adx / 50)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=current_adx,
            metadata={
                "adx": current_adx,
                "plus_di": current_plus_di,
                "minus_di": current_minus_di,
            },
        )


class IchimokuStrategy(ExpertStrategy):
    """
    TF-04: Ichimoku Cloud Strategy

    Entry: Price above cloud AND Tenkan > Kijun
    Exit: Opposite conditions

    Regime Hypothesis: Clear trending markets
    Failure Mode: Sideways, choppy markets where cloud is flat
    """

    def __init__(self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
        config = StrategyConfig(
            name="Ichimoku",
            category="trend_following",
            parameters={"tenkan": tenkan, "kijun": kijun, "senkou_b": senkou_b},
            regime_hypothesis="Clear trends with price outside cloud",
            failure_mode="Sideways markets with price inside cloud",
        )
        super().__init__(config)
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b

    def _donchian_mid(self, high: pd.Series, low: pd.Series, period: int) -> float:
        """Calculate Donchian midpoint."""
        return (high.iloc[-period:].max() + low.iloc[-period:].min()) / 2

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.senkou_b + self.kijun:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        current_price = close.iloc[-1]

        # Tenkan-sen (Conversion Line)
        tenkan_sen = self._donchian_mid(high, low, self.tenkan)

        # Kijun-sen (Base Line)
        kijun_sen = self._donchian_mid(high, low, self.kijun)

        # Senkou Span A (Leading Span A) - shifted forward 26 periods, so use historical
        senkou_a = (tenkan_sen + kijun_sen) / 2

        # Senkou Span B (Leading Span B) - shifted forward 26 periods
        senkou_b = self._donchian_mid(
            high.iloc[:-self.kijun] if len(high) > self.kijun else high,
            low.iloc[:-self.kijun] if len(low) > self.kijun else low,
            self.senkou_b
        )

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Determine signal
        if current_price > cloud_top and tenkan_sen > kijun_sen:
            signal = Signal.LONG
            confidence = min(1.0, (current_price - cloud_top) / cloud_top * 20 + 0.6)
        elif current_price < cloud_bottom and tenkan_sen < kijun_sen:
            signal = Signal.SHORT
            confidence = min(1.0, (cloud_bottom - current_price) / cloud_bottom * 20 + 0.6)
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=current_price / kijun_sen,
            metadata={
                "tenkan": tenkan_sen,
                "kijun": kijun_sen,
                "cloud_top": cloud_top,
                "cloud_bottom": cloud_bottom,
            },
        )


class ParabolicSARStrategy(ExpertStrategy):
    """
    TF-05: Parabolic SAR Strategy

    Entry: SAR below price (long) or above price (short)
    Exit: SAR flips

    Regime Hypothesis: Directional trending markets
    Failure Mode: Consolidation periods with frequent SAR flips
    """

    def __init__(self, af_start: float = 0.02, af_max: float = 0.2):
        config = StrategyConfig(
            name="ParabolicSAR",
            category="trend_following",
            parameters={"af_start": af_start, "af_max": af_max},
            regime_hypothesis="Directional moves with clear momentum",
            failure_mode="Tight consolidations, frequent reversals",
        )
        super().__init__(config)
        self.af_start = af_start
        self.af_max = af_max

    def _calculate_sar(self, high: pd.Series, low: pd.Series) -> tuple:
        """Calculate Parabolic SAR."""
        n = len(high)
        sar = np.zeros(n)
        trend = np.zeros(n)  # 1 = up, -1 = down
        ep = np.zeros(n)  # Extreme Point
        af = np.zeros(n)  # Acceleration Factor

        # Initialize
        trend[0] = 1
        sar[0] = low.iloc[0]
        ep[0] = high.iloc[0]
        af[0] = self.af_start

        for i in range(1, n):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = min(sar[i], low.iloc[i-1], low.iloc[max(0, i-2)])

                if low.iloc[i] < sar[i]:  # Reversal
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low.iloc[i]
                    af[i] = self.af_start
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + self.af_start, self.af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                sar[i] = max(sar[i], high.iloc[i-1], high.iloc[max(0, i-2)])

                if high.iloc[i] > sar[i]:  # Reversal
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high.iloc[i]
                    af[i] = self.af_start
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + self.af_start, self.af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]

        return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < 20:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        sar, trend = self._calculate_sar(high, low)
        current_sar = sar.iloc[-1]
        current_trend = trend.iloc[-1]
        current_price = close.iloc[-1]

        distance = abs(current_price - current_sar) / current_price

        if current_trend == 1:  # SAR below price
            signal = Signal.LONG
            confidence = min(1.0, distance * 20 + 0.5)
        else:  # SAR above price
            signal = Signal.SHORT
            confidence = min(1.0, distance * 20 + 0.5)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=current_sar,
            metadata={"sar": current_sar, "trend": current_trend},
        )


class DonchianBreakoutStrategy(ExpertStrategy):
    """
    TF-06: Donchian Channel Breakout Strategy (Turtle Trading)

    Entry: Price breaks 20-day high (long) or 20-day low (short)
    Exit: Price breaks opposite channel

    Regime Hypothesis: Breakout/trending environments
    Failure Mode: Range-bound markets with false breakouts
    """

    def __init__(self, entry_period: int = 20, exit_period: int = 10):
        config = StrategyConfig(
            name="DonchianBreakout",
            category="trend_following",
            parameters={"entry_period": entry_period, "exit_period": exit_period},
            regime_hypothesis="Breakout markets, new highs/lows",
            failure_mode="False breakouts in ranging markets",
        )
        super().__init__(config)
        self.entry_period = entry_period
        self.exit_period = exit_period
        self._position = Signal.CASH

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.entry_period:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {"reason": "warmup"})

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        current_price = close.iloc[-1]

        # Entry channels (exclude current bar)
        entry_high = high.iloc[-self.entry_period - 1:-1].max()
        entry_low = low.iloc[-self.entry_period - 1:-1].min()

        # Exit channels
        exit_high = high.iloc[-self.exit_period - 1:-1].max()
        exit_low = low.iloc[-self.exit_period - 1:-1].min()

        # Determine signal
        if current_price > entry_high:
            signal = Signal.LONG
            confidence = min(1.0, (current_price - entry_high) / entry_high * 50 + 0.6)
        elif current_price < entry_low:
            signal = Signal.SHORT
            confidence = min(1.0, (entry_low - current_price) / entry_low * 50 + 0.6)
        else:
            signal = Signal.CASH
            confidence = 0.3

        channel_width = (entry_high - entry_low) / current_price

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=channel_width,
            metadata={
                "entry_high": entry_high,
                "entry_low": entry_low,
                "exit_high": exit_high,
                "exit_low": exit_low,
            },
        )

"""
Micro-Regime Detection System
=============================

Granular, multi-dimensional regime classification for tactical trading.

Based on Expert Panel recommendations:
- 5-level trend states
- 4-level volatility states
- 3-level momentum states
- 3-level mean-reversion states

This creates up to 180 theoretical micro-regimes, with ~30-50 commonly occurring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np


class TrendState(Enum):
    """5-level trend classification."""
    STRONG_BULL = 2
    BULL = 1
    NEUTRAL = 0
    BEAR = -1
    STRONG_BEAR = -2


class VolatilityState(Enum):
    """4-level volatility classification."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRISIS = 3


class MomentumState(Enum):
    """3-level momentum acceleration classification."""
    ACCELERATING = 1
    STEADY = 0
    DECELERATING = -1


class MeanReversionState(Enum):
    """3-level overbought/oversold classification."""
    OVERBOUGHT = 1
    NEUTRAL = 0
    OVERSOLD = -1


@dataclass
class MicroRegime:
    """Complete micro-regime state."""
    trend: TrendState
    volatility: VolatilityState
    momentum: MomentumState
    mean_reversion: MeanReversionState

    @property
    def code(self) -> str:
        """Short code for this regime."""
        trend_map = {
            TrendState.STRONG_BULL: "SB",
            TrendState.BULL: "B",
            TrendState.NEUTRAL: "N",
            TrendState.BEAR: "D",  # D for Down
            TrendState.STRONG_BEAR: "SD",
        }
        vol_map = {
            VolatilityState.LOW: "L",
            VolatilityState.NORMAL: "N",
            VolatilityState.HIGH: "H",
            VolatilityState.CRISIS: "C",
        }
        mom_map = {
            MomentumState.ACCELERATING: "+",
            MomentumState.STEADY: "=",
            MomentumState.DECELERATING: "-",
        }
        mr_map = {
            MeanReversionState.OVERBOUGHT: "O",
            MeanReversionState.NEUTRAL: "N",
            MeanReversionState.OVERSOLD: "S",
        }
        return f"{trend_map[self.trend]}{vol_map[self.volatility]}{mom_map[self.momentum]}{mr_map[self.mean_reversion]}"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return f"{self.trend.name} | {self.volatility.name}_VOL | MOM_{self.momentum.name} | {self.mean_reversion.name}"

    def __hash__(self):
        return hash((self.trend, self.volatility, self.momentum, self.mean_reversion))

    def __eq__(self, other):
        if not isinstance(other, MicroRegime):
            return False
        return (self.trend == other.trend and
                self.volatility == other.volatility and
                self.momentum == other.momentum and
                self.mean_reversion == other.mean_reversion)


class MicroRegimeDetector:
    """
    Multi-dimensional micro-regime detector.

    Detects regimes on tactical timeframes (days to weeks) rather than
    secular trends (years).

    Parameters:
        trend_lookback: Days for trend calculation (default 20)
        vol_lookback: Days for volatility calculation (default 20)
        mom_short: Short momentum window (default 5)
        mom_long: Long momentum window (default 20)
        rsi_period: RSI calculation period (default 14)
        bb_period: Bollinger Band period (default 20)
        bb_std: Bollinger Band standard deviations (default 2.0)
    """

    def __init__(
        self,
        trend_lookback: int = 20,
        vol_lookback: int = 20,
        mom_short: int = 5,
        mom_long: int = 20,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        # Trend thresholds (annualized)
        strong_trend_threshold: float = 0.10,  # 10% in 20 days
        trend_threshold: float = 0.03,  # 3% in 20 days
        # Volatility thresholds (annualized)
        low_vol_threshold: float = 0.10,  # 10% annualized
        high_vol_threshold: float = 0.20,  # 20% annualized
        crisis_vol_threshold: float = 0.35,  # 35% annualized
        # RSI thresholds
        overbought_rsi: float = 70.0,
        oversold_rsi: float = 30.0,
    ):
        self.trend_lookback = trend_lookback
        self.vol_lookback = vol_lookback
        self.mom_short = mom_short
        self.mom_long = mom_long
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std

        self.strong_trend_threshold = strong_trend_threshold
        self.trend_threshold = trend_threshold
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.crisis_vol_threshold = crisis_vol_threshold
        self.overbought_rsi = overbought_rsi
        self.oversold_rsi = oversold_rsi

    def detect(self, data: pd.DataFrame, idx: int) -> Optional[MicroRegime]:
        """
        Detect micro-regime at a specific index.

        Uses only data up to and including idx (no look-ahead).

        Args:
            data: DataFrame with OHLCV data
            idx: Current index (integer position)

        Returns:
            MicroRegime object or None if insufficient data
        """
        # Minimum data requirement
        min_required = max(
            self.trend_lookback,
            self.vol_lookback,
            self.mom_long,
            self.rsi_period + 1,
            self.bb_period
        ) + 10  # Buffer

        if idx < min_required:
            return None

        # Get data slice (no look-ahead)
        close = data["close"].iloc[:idx + 1]
        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]

        # Calculate indicators
        trend_state = self._detect_trend(close)
        vol_state = self._detect_volatility(close)
        mom_state = self._detect_momentum(close)
        mr_state = self._detect_mean_reversion(close, high, low)

        return MicroRegime(
            trend=trend_state,
            volatility=vol_state,
            momentum=mom_state,
            mean_reversion=mr_state
        )

    def _detect_trend(self, close: pd.Series) -> TrendState:
        """Detect 5-level trend state."""
        # 20-day return
        ret_20d = (close.iloc[-1] / close.iloc[-self.trend_lookback] - 1)

        # Price vs moving averages
        sma50 = close.iloc[-50:].mean() if len(close) >= 50 else close.mean()
        sma200 = close.iloc[-200:].mean() if len(close) >= 200 else close.mean()
        current_price = close.iloc[-1]

        above_sma50 = current_price > sma50
        above_sma200 = current_price > sma200

        # Strong bull: >10% return AND above both MAs
        if ret_20d > self.strong_trend_threshold and above_sma50 and above_sma200:
            return TrendState.STRONG_BULL

        # Strong bear: <-10% return AND below both MAs
        if ret_20d < -self.strong_trend_threshold and not above_sma50 and not above_sma200:
            return TrendState.STRONG_BEAR

        # Bull: >3% return OR (above SMA50 and positive return)
        if ret_20d > self.trend_threshold or (above_sma50 and ret_20d > 0):
            return TrendState.BULL

        # Bear: <-3% return OR (below SMA50 and negative return)
        if ret_20d < -self.trend_threshold or (not above_sma50 and ret_20d < 0):
            return TrendState.BEAR

        return TrendState.NEUTRAL

    def _detect_volatility(self, close: pd.Series) -> VolatilityState:
        """Detect 4-level volatility state."""
        # Calculate 20-day realized volatility (annualized)
        returns = close.pct_change().iloc[-self.vol_lookback:]
        realized_vol = returns.std() * np.sqrt(252)

        if realized_vol < self.low_vol_threshold:
            return VolatilityState.LOW
        elif realized_vol < self.high_vol_threshold:
            return VolatilityState.NORMAL
        elif realized_vol < self.crisis_vol_threshold:
            return VolatilityState.HIGH
        else:
            return VolatilityState.CRISIS

    def _detect_momentum(self, close: pd.Series) -> MomentumState:
        """Detect 3-level momentum acceleration state."""
        # Short-term vs long-term momentum
        ret_short = (close.iloc[-1] / close.iloc[-self.mom_short] - 1)
        ret_long = (close.iloc[-1] / close.iloc[-self.mom_long] - 1)

        # Normalize long-term to short-term timeframe
        ret_long_normalized = ret_long * (self.mom_short / self.mom_long)

        # Acceleration threshold: short > 1.5x normalized long
        if ret_short > ret_long_normalized * 1.5 and ret_short > 0:
            return MomentumState.ACCELERATING

        # Deceleration: short < 0.5x normalized long OR reversing
        if ret_short < ret_long_normalized * 0.5 or (ret_short < 0 and ret_long > 0):
            return MomentumState.DECELERATING

        return MomentumState.STEADY

    def _detect_mean_reversion(
        self, close: pd.Series, high: pd.Series, low: pd.Series
    ) -> MeanReversionState:
        """Detect 3-level mean-reversion state (overbought/oversold)."""
        # Calculate RSI
        rsi = self._calculate_rsi(close, self.rsi_period)

        # Calculate Bollinger Bands
        sma = close.iloc[-self.bb_period:].mean()
        std = close.iloc[-self.bb_period:].std()
        upper_bb = sma + self.bb_std * std
        lower_bb = sma - self.bb_std * std
        current_price = close.iloc[-1]

        # Overbought: RSI > 70 OR price > upper BB
        if rsi > self.overbought_rsi or current_price > upper_bb:
            return MeanReversionState.OVERBOUGHT

        # Oversold: RSI < 30 OR price < lower BB
        if rsi < self.oversold_rsi or current_price < lower_bb:
            return MeanReversionState.OVERSOLD

        return MeanReversionState.NEUTRAL

    def _calculate_rsi(self, close: pd.Series, period: int) -> float:
        """Calculate RSI."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.iloc[-period:].mean()
        avg_loss = loss.iloc[-period:].mean()

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def label_history(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Label entire historical dataset with micro-regimes.

        Returns DataFrame with regime columns added.
        """
        n = len(data)

        # Initialize arrays
        trend_states = []
        vol_states = []
        mom_states = []
        mr_states = []
        regime_codes = []

        for idx in range(n):
            regime = self.detect(data, idx)

            if regime is None:
                trend_states.append(None)
                vol_states.append(None)
                mom_states.append(None)
                mr_states.append(None)
                regime_codes.append(None)
            else:
                trend_states.append(regime.trend.name)
                vol_states.append(regime.volatility.name)
                mom_states.append(regime.momentum.name)
                mr_states.append(regime.mean_reversion.name)
                regime_codes.append(regime.code)

        result = data.copy()
        result["trend_state"] = trend_states
        result["vol_state"] = vol_states
        result["mom_state"] = mom_states
        result["mr_state"] = mr_states
        result["micro_regime"] = regime_codes

        return result

    def analyze_regime_distribution(
        self, labeled_data: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Analyze distribution of micro-regimes in labeled data.

        Returns statistics about regime frequency, duration, and transitions.
        """
        # Filter to valid regimes
        valid_data = labeled_data[labeled_data["micro_regime"].notna()].copy()

        if len(valid_data) == 0:
            return {"error": "No valid regime data"}

        # Regime frequency
        regime_counts = valid_data["micro_regime"].value_counts()
        regime_pcts = regime_counts / len(valid_data) * 100

        # Regime durations
        regime_durations = self._calculate_regime_durations(valid_data)

        # Component distributions
        trend_dist = valid_data["trend_state"].value_counts(normalize=True) * 100
        vol_dist = valid_data["vol_state"].value_counts(normalize=True) * 100
        mom_dist = valid_data["mom_state"].value_counts(normalize=True) * 100
        mr_dist = valid_data["mr_state"].value_counts(normalize=True) * 100

        # Transition matrix (simplified: trend transitions)
        trend_transitions = self._calculate_transitions(valid_data, "trend_state")

        return {
            "total_days": len(valid_data),
            "unique_regimes": len(regime_counts),
            "regime_counts": regime_counts.to_dict(),
            "regime_percentages": regime_pcts.to_dict(),
            "avg_regime_duration": regime_durations["avg"],
            "median_regime_duration": regime_durations["median"],
            "max_regime_duration": regime_durations["max"],
            "trend_distribution": trend_dist.to_dict(),
            "volatility_distribution": vol_dist.to_dict(),
            "momentum_distribution": mom_dist.to_dict(),
            "mean_reversion_distribution": mr_dist.to_dict(),
            "trend_transition_matrix": trend_transitions,
        }

    def _calculate_regime_durations(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate regime duration statistics."""
        regimes = data["micro_regime"].values
        durations = []
        current_regime = regimes[0]
        current_duration = 1

        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_regime = regimes[i]
                current_duration = 1

        durations.append(current_duration)

        return {
            "avg": np.mean(durations),
            "median": np.median(durations),
            "max": max(durations),
            "min": min(durations),
            "std": np.std(durations),
        }

    def _calculate_transitions(
        self, data: pd.DataFrame, column: str
    ) -> Dict[str, Dict[str, float]]:
        """Calculate transition probabilities."""
        states = data[column].values
        transitions = {}

        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]

            if from_state not in transitions:
                transitions[from_state] = {}

            if to_state not in transitions[from_state]:
                transitions[from_state][to_state] = 0

            transitions[from_state][to_state] += 1

        # Normalize to probabilities
        for from_state in transitions:
            total = sum(transitions[from_state].values())
            for to_state in transitions[from_state]:
                transitions[from_state][to_state] /= total

        return transitions


def get_regime_for_strategy(micro_regime: MicroRegime) -> Dict[str, float]:
    """
    Get recommended strategy weights for a given micro-regime.

    Returns dictionary of strategy category -> weight.

    This encodes expert knowledge about which strategies work in which conditions.
    """
    weights = {
        "trend_following": 0.0,
        "mean_reversion": 0.0,
        "volatility": 0.0,
        "momentum": 0.0,
        "defensive": 0.0,
    }

    # Trend-following works in strong trends with steady/accelerating momentum
    if micro_regime.trend in [TrendState.STRONG_BULL, TrendState.STRONG_BEAR]:
        if micro_regime.momentum != MomentumState.DECELERATING:
            weights["trend_following"] = 0.5
    elif micro_regime.trend in [TrendState.BULL, TrendState.BEAR]:
        if micro_regime.momentum == MomentumState.ACCELERATING:
            weights["trend_following"] = 0.4
        elif micro_regime.momentum == MomentumState.STEADY:
            weights["trend_following"] = 0.3

    # Mean-reversion works in low/normal volatility with overbought/oversold conditions
    if micro_regime.volatility in [VolatilityState.LOW, VolatilityState.NORMAL]:
        if micro_regime.mean_reversion == MeanReversionState.OVERBOUGHT:
            weights["mean_reversion"] = 0.3
        elif micro_regime.mean_reversion == MeanReversionState.OVERSOLD:
            weights["mean_reversion"] = 0.3

    # Volatility strategies work in high/crisis vol
    if micro_regime.volatility in [VolatilityState.HIGH, VolatilityState.CRISIS]:
        weights["volatility"] = 0.4

    # Momentum works in accelerating conditions
    if micro_regime.momentum == MomentumState.ACCELERATING:
        weights["momentum"] = 0.2

    # Defensive in crisis vol or decelerating strong trends
    if micro_regime.volatility == VolatilityState.CRISIS:
        weights["defensive"] = 0.3
    elif (micro_regime.trend in [TrendState.STRONG_BEAR] and
          micro_regime.momentum == MomentumState.ACCELERATING):  # Accelerating down
        weights["defensive"] = 0.4

    # Normalize weights
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        # Default: equal weight defensive
        weights["defensive"] = 1.0

    return weights

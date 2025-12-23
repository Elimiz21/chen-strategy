"""
Regime Detection Implementations
================================

Four approaches based on Phase 1 Regime Detection Survey:
1. Rules-Based: SMA crossovers + volatility thresholds
2. Threshold: Volatility percentiles + momentum
3. HMM: Hidden Markov Model with walk-forward training
4. Hybrid: Rules primary + ML confidence scoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np


class Regime(Enum):
    """Market regime states."""
    BULL_LOW_VOL = "BULL_LOW_VOL"
    BULL_NORMAL_VOL = "BULL_NORMAL_VOL"
    BULL_HIGH_VOL = "BULL_HIGH_VOL"
    BEAR_LOW_VOL = "BEAR_LOW_VOL"
    BEAR_NORMAL_VOL = "BEAR_NORMAL_VOL"
    BEAR_HIGH_VOL = "BEAR_HIGH_VOL"
    TRANSITION = "TRANSITION"


@dataclass
class RegimeResult:
    """Result of regime detection."""
    regime: Regime
    trend: str  # BULL, BEAR, TRANSITION
    volatility: str  # LOW, NORMAL, HIGH
    confidence: float  # 0.0 to 1.0
    probabilities: Optional[dict] = None  # For probabilistic models


class RegimeDetector(ABC):
    """Base class for regime detectors."""

    @abstractmethod
    def detect(self, data: pd.DataFrame, idx: int) -> RegimeResult:
        """
        Detect regime at a specific point in time.

        Args:
            data: DataFrame with OHLCV data
            idx: Index position (uses only data up to idx, no look-ahead)

        Returns:
            RegimeResult with regime classification
        """
        pass

    def detect_all(self, data: pd.DataFrame, start_idx: int = 200) -> pd.Series:
        """
        Detect regimes for all points in the data.

        Returns:
            Series of Regime values indexed by date
        """
        regimes = []
        for idx in range(start_idx, len(data)):
            result = self.detect(data, idx)
            regimes.append(result.regime.value)

        # Pad beginning with TRANSITION
        padding = [Regime.TRANSITION.value] * start_idx
        all_regimes = padding + regimes

        return pd.Series(all_regimes, index=data.index, name="regime")


class RulesBasedDetector(RegimeDetector):
    """
    Rules-Based Regime Detection (Approach 1)

    Uses moving average relationships for trend and ATR ratio for volatility.

    Trend Rules:
        BULL: Price > 200 SMA AND 50 SMA > 200 SMA
        BEAR: Price < 200 SMA AND 50 SMA < 200 SMA
        TRANSITION: Otherwise

    Volatility Rules:
        HIGH: ATR(20) / ATR(60) > 1.5
        LOW: ATR(20) / ATR(60) < 0.75
        NORMAL: Otherwise
    """

    def __init__(
        self,
        fast_sma: int = 50,
        slow_sma: int = 200,
        atr_short: int = 20,
        atr_long: int = 60,
        vol_high_threshold: float = 1.5,
        vol_low_threshold: float = 0.75,
    ):
        self.fast_sma = fast_sma
        self.slow_sma = slow_sma
        self.atr_short = atr_short
        self.atr_long = atr_long
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Calculate ATR for a given period."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.iloc[-period:].mean()

    def detect(self, data: pd.DataFrame, idx: int) -> RegimeResult:
        if idx < self.slow_sma:
            return RegimeResult(
                regime=Regime.TRANSITION,
                trend="TRANSITION",
                volatility="NORMAL",
                confidence=0.0,
            )

        close = data["close"].iloc[:idx + 1]
        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]

        current_price = close.iloc[-1]

        # Calculate SMAs
        sma_fast = close.iloc[-self.fast_sma:].mean()
        sma_slow = close.iloc[-self.slow_sma:].mean()

        # Trend detection
        if current_price > sma_slow and sma_fast > sma_slow:
            trend = "BULL"
            trend_strength = (current_price / sma_slow - 1) + (sma_fast / sma_slow - 1)
        elif current_price < sma_slow and sma_fast < sma_slow:
            trend = "BEAR"
            trend_strength = (1 - current_price / sma_slow) + (1 - sma_fast / sma_slow)
        else:
            trend = "TRANSITION"
            trend_strength = 0.0

        # Volatility detection
        if idx >= self.atr_long:
            atr_short = self._calculate_atr(high, low, close, self.atr_short)
            atr_long = self._calculate_atr(high, low, close, self.atr_long)
            vol_ratio = atr_short / atr_long if atr_long > 0 else 1.0

            if vol_ratio > self.vol_high_threshold:
                volatility = "HIGH"
            elif vol_ratio < self.vol_low_threshold:
                volatility = "LOW"
            else:
                volatility = "NORMAL"
        else:
            volatility = "NORMAL"
            vol_ratio = 1.0

        # Combine into regime
        if trend == "TRANSITION":
            regime = Regime.TRANSITION
        else:
            regime_name = f"{trend}_{volatility}_VOL"
            regime = Regime[regime_name]

        # Confidence based on trend strength
        confidence = min(1.0, abs(trend_strength) * 5 + 0.3)

        return RegimeResult(
            regime=regime,
            trend=trend,
            volatility=volatility,
            confidence=confidence,
        )


class ThresholdDetector(RegimeDetector):
    """
    Threshold-Based Regime Detection (Approach 3)

    Uses volatility percentiles and momentum for regime classification.
    """

    def __init__(
        self,
        vol_window: int = 20,
        vol_lookback: int = 252,
        momentum_window: int = 252,
        momentum_skip: int = 21,
    ):
        self.vol_window = vol_window
        self.vol_lookback = vol_lookback
        self.momentum_window = momentum_window
        self.momentum_skip = momentum_skip

    def detect(self, data: pd.DataFrame, idx: int) -> RegimeResult:
        if idx < max(self.vol_lookback, self.momentum_window):
            return RegimeResult(
                regime=Regime.TRANSITION,
                trend="TRANSITION",
                volatility="NORMAL",
                confidence=0.0,
            )

        close = data["close"].iloc[:idx + 1]
        returns = close.pct_change()

        # Realized volatility
        current_vol = returns.iloc[-self.vol_window:].std() * np.sqrt(252)
        historical_vols = returns.rolling(self.vol_window).std() * np.sqrt(252)
        vol_percentile = (historical_vols.iloc[-self.vol_lookback:] < current_vol).mean()

        # Volatility regime
        if vol_percentile > 0.8:
            volatility = "HIGH"
        elif vol_percentile < 0.2:
            volatility = "LOW"
        else:
            volatility = "NORMAL"

        # Momentum (12-1)
        start_price = close.iloc[-self.momentum_window]
        end_price = close.iloc[-self.momentum_skip]
        momentum = (end_price / start_price) - 1

        # Trend from momentum
        if momentum > 0.05:  # 5% threshold
            trend = "BULL"
        elif momentum < -0.05:
            trend = "BEAR"
        else:
            trend = "TRANSITION"

        # Combine
        if trend == "TRANSITION":
            regime = Regime.TRANSITION
        else:
            regime_name = f"{trend}_{volatility}_VOL"
            regime = Regime[regime_name]

        confidence = min(1.0, abs(momentum) * 2 + 0.4)

        return RegimeResult(
            regime=regime,
            trend=trend,
            volatility=volatility,
            confidence=confidence,
            probabilities={"vol_percentile": vol_percentile, "momentum": momentum},
        )


class HMMDetector(RegimeDetector):
    """
    Hidden Markov Model Regime Detection (Approach 2)

    Uses HMM with walk-forward training to avoid look-ahead bias.
    Requires hmmlearn library.
    """

    def __init__(
        self,
        n_states: int = 3,
        train_window: int = 252 * 5,  # 5 years
        retrain_frequency: int = 21,  # Monthly
    ):
        self.n_states = n_states
        self.train_window = train_window
        self.retrain_frequency = retrain_frequency
        self._model = None
        self._last_train_idx = 0
        self._state_mapping = {}

    def _train_model(self, returns: np.ndarray):
        """Train HMM on historical returns."""
        try:
            from hmmlearn import hmm
        except ImportError:
            raise ImportError("hmmlearn required for HMM detector. Install with: pip install hmmlearn")

        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )
        model.fit(returns.reshape(-1, 1))
        self._model = model

        # Map states by mean return (highest mean = bull, lowest = bear)
        means = model.means_.flatten()
        sorted_indices = np.argsort(means)

        self._state_mapping = {
            sorted_indices[0]: "BEAR",  # Lowest mean
            sorted_indices[-1]: "BULL",  # Highest mean
        }
        for i in sorted_indices[1:-1]:
            self._state_mapping[i] = "TRANSITION"

    def detect(self, data: pd.DataFrame, idx: int) -> RegimeResult:
        if idx < self.train_window:
            return RegimeResult(
                regime=Regime.TRANSITION,
                trend="TRANSITION",
                volatility="NORMAL",
                confidence=0.0,
            )

        close = data["close"].iloc[:idx + 1]
        returns = close.pct_change().dropna().values

        # Retrain if needed
        if self._model is None or (idx - self._last_train_idx) >= self.retrain_frequency:
            train_returns = returns[-self.train_window:]
            self._train_model(train_returns)
            self._last_train_idx = idx

        # Predict current state
        recent_returns = returns[-20:]  # Use recent window for prediction
        state = self._model.predict(recent_returns.reshape(-1, 1))[-1]
        proba = self._model.predict_proba(recent_returns.reshape(-1, 1))[-1]

        trend = self._state_mapping.get(state, "TRANSITION")

        # Volatility from recent returns
        vol = np.std(recent_returns) * np.sqrt(252)
        if vol > 0.30:
            volatility = "HIGH"
        elif vol < 0.15:
            volatility = "LOW"
        else:
            volatility = "NORMAL"

        # Combine
        if trend == "TRANSITION":
            regime = Regime.TRANSITION
        else:
            regime_name = f"{trend}_{volatility}_VOL"
            regime = Regime[regime_name]

        confidence = float(proba.max())

        return RegimeResult(
            regime=regime,
            trend=trend,
            volatility=volatility,
            confidence=confidence,
            probabilities={f"state_{i}": float(p) for i, p in enumerate(proba)},
        )


class HybridDetector(RegimeDetector):
    """
    Hybrid Regime Detection (Approach 4)

    Uses rules-based detection as primary signal with optional ML confidence scoring.
    Implements "tilt not switch" behavior.
    """

    def __init__(
        self,
        rules_weight: float = 0.6,
        ml_weight: float = 0.4,
        use_ml: bool = True,
    ):
        self.rules_weight = rules_weight
        self.ml_weight = ml_weight
        self.use_ml = use_ml

        self._rules_detector = RulesBasedDetector()
        self._threshold_detector = ThresholdDetector()

    def detect(self, data: pd.DataFrame, idx: int) -> RegimeResult:
        # Primary: Rules-based
        rules_result = self._rules_detector.detect(data, idx)

        if not self.use_ml or idx < 252:
            return rules_result

        # Secondary: Threshold-based for confirmation
        threshold_result = self._threshold_detector.detect(data, idx)

        # Combine confidences
        if rules_result.trend == threshold_result.trend:
            # Agreement - boost confidence
            combined_confidence = min(1.0, rules_result.confidence * 1.2)
        else:
            # Disagreement - reduce confidence
            combined_confidence = rules_result.confidence * 0.7

        # Weight the confidences
        final_confidence = (
            self.rules_weight * rules_result.confidence +
            self.ml_weight * threshold_result.confidence
        )

        return RegimeResult(
            regime=rules_result.regime,
            trend=rules_result.trend,
            volatility=rules_result.volatility,
            confidence=min(1.0, final_confidence),
            probabilities={
                "rules_confidence": rules_result.confidence,
                "threshold_confidence": threshold_result.confidence,
                "agreement": rules_result.trend == threshold_result.trend,
            },
        )

    def get_position_multiplier(self, result: RegimeResult) -> float:
        """
        Get position multiplier for "tilt not switch" behavior.

        Returns:
            Multiplier between 0.0 and 1.0
        """
        if result.confidence > 0.8:
            return 1.0
        elif result.confidence > 0.6:
            return 0.7
        elif result.confidence > 0.4:
            return 0.4
        else:
            return 0.0


def compute_regime_transition_matrix(regimes: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """
    Compute regime transition probability matrix.

    Args:
        regimes: Series of regime labels

    Returns:
        Tuple of (transition matrix, state labels)
    """
    from collections import Counter

    transitions = Counter(zip(regimes.iloc[:-1], regimes.iloc[1:]))
    states = sorted(set(regimes.dropna()))
    n_states = len(states)

    matrix = np.zeros((n_states, n_states))
    state_to_idx = {s: i for i, s in enumerate(states)}

    for (from_state, to_state), count in transitions.items():
        if from_state in state_to_idx and to_state in state_to_idx:
            matrix[state_to_idx[from_state], state_to_idx[to_state]] = count

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums

    return matrix, states


def analyze_regime_persistence(regimes: pd.Series) -> dict:
    """
    Analyze regime persistence statistics.

    Returns:
        Dictionary with persistence metrics per regime
    """
    results = {}

    for regime in regimes.unique():
        if pd.isna(regime):
            continue

        # Find consecutive runs
        is_regime = (regimes == regime).astype(int)
        changes = is_regime.diff().fillna(0)

        run_lengths = []
        current_run = 0

        for val, change in zip(is_regime, changes):
            if val == 1:
                current_run += 1
            elif current_run > 0:
                run_lengths.append(current_run)
                current_run = 0

        if current_run > 0:
            run_lengths.append(current_run)

        if run_lengths:
            results[regime] = {
                "count": len(run_lengths),
                "avg_duration": np.mean(run_lengths),
                "min_duration": np.min(run_lengths),
                "max_duration": np.max(run_lengths),
                "total_days": sum(run_lengths),
                "pct_of_time": sum(run_lengths) / len(regimes) * 100,
            }

    return results

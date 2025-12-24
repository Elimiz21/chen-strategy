"""
Sophisticated Academic Baselines
================================

Based on Expert Panel recommendations, these baselines represent
state-of-the-art systematic strategies from academic literature.

These replace the naive baselines (Buy & Hold, SMA200, Golden Cross)
with proper benchmarks that any new strategy must beat.

References:
- Moskowitz, Ooi, Pedersen (2012): "Time Series Momentum"
- Asness, Frazzini, Pedersen (2012): "Leverage Aversion and Risk Parity"
- Barroso, Santa-Clara (2015): "Momentum Has Its Moments"
- Daniel, Moskowitz (2016): "Momentum Crashes"
"""

from typing import Tuple
import pandas as pd
import numpy as np

from .base import ExpertStrategy, Signal, StrategyResult, StrategyConfig


class VolatilityTargetingBaseline(ExpertStrategy):
    """
    Risk Parity / Constant Volatility Target Baseline.

    Academic foundation: Asness, Frazzini, Pedersen (2012)
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        vol_lookback: int = 20,
        rebalance_days: int = 5,
        vol_floor: float = 0.05,
        vol_cap: float = 0.50,
    ):
        config = StrategyConfig(
            name="VolTargetAcademic",
            category="academic_baseline",
            parameters={"target_vol": target_vol, "vol_lookback": vol_lookback},
            regime_hypothesis="Works in all regimes by adapting to volatility",
            failure_mode="May underperform in low-vol trending markets",
        )
        super().__init__(config)
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap
        self._last_rebalance_idx = -999

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.vol_lookback + 1:
            return StrategyResult(
                signal=Signal.CASH,
                confidence=0.0,
                raw_indicator=0.0,
                metadata={"reason": "Insufficient data"}
            )

        if idx - self._last_rebalance_idx < self.rebalance_days:
            return StrategyResult(
                signal=Signal.LONG,
                confidence=0.5,
                raw_indicator=0.0,
                metadata={"reason": "Not rebalance day"}
            )

        self._last_rebalance_idx = idx

        close = data["close"].iloc[idx - self.vol_lookback:idx + 1]
        returns = close.pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252)
        realized_vol = max(self.vol_floor, min(self.vol_cap, realized_vol))
        target_leverage = self.target_vol / realized_vol

        return StrategyResult(
            signal=Signal.LONG,
            confidence=min(1.0, target_leverage),
            raw_indicator=realized_vol,
            metadata={"realized_vol": realized_vol, "target_leverage": target_leverage}
        )

    def get_position_size(self, signal: Signal, confidence: float, max_leverage: float = 2.0) -> float:
        if signal == Signal.CASH:
            return 0.0
        return min(confidence, max_leverage)


class TimeSeriesMomentumBaseline(ExpertStrategy):
    """
    Time-Series Momentum (TSMOM) Baseline.

    Academic foundation: Moskowitz, Ooi, Pedersen (2012)
    """

    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        vol_lookback: int = 20,
        target_vol: float = 0.15,
    ):
        config = StrategyConfig(
            name="TSMOM",
            category="academic_baseline",
            parameters={"lookback_months": lookback_months},
            regime_hypothesis="Works in trending markets",
            failure_mode="Whipsaws in choppy markets",
        )
        super().__init__(config)
        self.lookback_days = lookback_months * 21
        self.skip_days = skip_months * 21
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        required = self.lookback_days + self.skip_days + 1
        if idx < required:
            return StrategyResult(
                signal=Signal.CASH,
                confidence=0.0,
                raw_indicator=0.0,
                metadata={"reason": "Insufficient data"}
            )

        close = data["close"].iloc[:idx + 1]
        price_now = close.iloc[-1 - self.skip_days]
        price_12m_ago = close.iloc[-self.lookback_days - self.skip_days]
        momentum_return = (price_now / price_12m_ago) - 1

        returns = close.pct_change().iloc[-self.vol_lookback:]
        realized_vol = returns.std() * np.sqrt(252)
        vol_scalar = self.target_vol / realized_vol if realized_vol > 0.05 else 1.0

        if momentum_return > 0:
            signal = Signal.LONG
        else:
            signal = Signal.SHORT

        confidence = min(1.0, abs(momentum_return) * vol_scalar * 2)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=momentum_return,
            metadata={"momentum_return": momentum_return, "vol_scalar": vol_scalar}
        )


class VolatilityRiskPremiumBaseline(ExpertStrategy):
    """Volatility Risk Premium (VRP) Baseline."""

    def __init__(
        self,
        short_vol_lookback: int = 5,
        long_vol_lookback: int = 60,
        signal_threshold: float = 0.02,
    ):
        config = StrategyConfig(
            name="VRP",
            category="academic_baseline",
            parameters={"short_vol_lookback": short_vol_lookback},
            regime_hypothesis="Exploits volatility risk premium",
            failure_mode="Vol regime changes",
        )
        super().__init__(config)
        self.short_vol_lookback = short_vol_lookback
        self.long_vol_lookback = long_vol_lookback
        self.signal_threshold = signal_threshold

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.long_vol_lookback + 1:
            return StrategyResult(
                signal=Signal.CASH,
                confidence=0.0,
                raw_indicator=0.0,
                metadata={"reason": "Insufficient data"}
            )

        close = data["close"].iloc[:idx + 1]
        returns = close.pct_change()

        short_vol = returns.iloc[-self.short_vol_lookback:].std() * np.sqrt(252)
        long_vol = returns.iloc[-self.long_vol_lookback:].std() * np.sqrt(252)
        vrp = long_vol - short_vol

        if vrp > self.signal_threshold:
            signal = Signal.LONG
            confidence = min(1.0, vrp / 0.10)
        elif vrp < -self.signal_threshold:
            signal = Signal.SHORT
            confidence = min(1.0, abs(vrp) / 0.10)
        else:
            signal = Signal.CASH
            confidence = 0.5

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=vrp,
            metadata={"short_vol": short_vol, "long_vol": long_vol, "vrp": vrp}
        )


class TrendFollowingEnsembleBaseline(ExpertStrategy):
    """Trend-Following Ensemble Baseline."""

    def __init__(
        self,
        lookbacks: Tuple[int, ...] = (10, 20, 50, 100, 200),
        vol_lookback: int = 20,
    ):
        config = StrategyConfig(
            name="TrendEnsemble",
            category="academic_baseline",
            parameters={"lookbacks": lookbacks},
            regime_hypothesis="Robust trend following",
            failure_mode="Choppy markets",
        )
        super().__init__(config)
        self.lookbacks = lookbacks
        self.vol_lookback = vol_lookback

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        max_lookback = max(self.lookbacks)
        if idx < max_lookback + 1:
            return StrategyResult(
                signal=Signal.CASH,
                confidence=0.0,
                raw_indicator=0.0,
                metadata={"reason": "Insufficient data"}
            )

        close = data["close"].iloc[:idx + 1]

        signals = []
        for lookback in self.lookbacks:
            if idx >= lookback:
                ret = (close.iloc[-1] / close.iloc[-lookback]) - 1
                normalized_ret = ret * np.sqrt(20 / lookback)
                signals.append(np.sign(normalized_ret))
            else:
                signals.append(0)

        avg_signal = np.mean(signals)
        agreement = abs(avg_signal)

        if avg_signal > 0.2:
            signal = Signal.LONG
            confidence = agreement
        elif avg_signal < -0.2:
            signal = Signal.SHORT
            confidence = agreement
        else:
            signal = Signal.CASH
            confidence = 0.3

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=avg_signal,
            metadata={"avg_signal": avg_signal, "agreement": agreement}
        )


class AdaptiveMomentumBaseline(ExpertStrategy):
    """Adaptive Momentum Baseline - Barroso, Santa-Clara (2015)."""

    def __init__(
        self,
        short_lookback: int = 21,
        long_lookback: int = 126,
        vol_threshold: float = 0.20,
        target_vol: float = 0.15,
    ):
        config = StrategyConfig(
            name="AdaptiveMomentum",
            category="academic_baseline",
            parameters={"vol_threshold": vol_threshold},
            regime_hypothesis="Crash-protected momentum",
            failure_mode="Rapid regime changes",
        )
        super().__init__(config)
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback
        self.vol_threshold = vol_threshold
        self.target_vol = target_vol

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < self.long_lookback + 20:
            return StrategyResult(
                signal=Signal.CASH,
                confidence=0.0,
                raw_indicator=0.0,
                metadata={"reason": "Insufficient data"}
            )

        close = data["close"].iloc[:idx + 1]
        returns = close.pct_change()
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)

        if current_vol > self.vol_threshold:
            lookback = self.short_lookback
            regime = "high_vol"
        else:
            lookback = self.long_lookback
            regime = "low_vol"

        momentum = (close.iloc[-1] / close.iloc[-lookback]) - 1

        mom_returns = []
        for i in range(20):
            if idx - i - lookback >= 0:
                past_mom = (close.iloc[-1 - i] / close.iloc[-1 - i - lookback]) - 1
                mom_returns.append(past_mom)
        mom_vol = np.std(mom_returns) if len(mom_returns) > 5 else 0.10
        vol_scalar = min(2.0, self.target_vol / max(mom_vol, 0.05))

        if momentum > 0:
            signal = Signal.LONG
        else:
            signal = Signal.SHORT

        confidence = min(1.0, abs(momentum) * vol_scalar)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=momentum,
            metadata={"momentum": momentum, "regime": regime, "lookback_used": lookback}
        )


class RiskOnRiskOffBaseline(ExpertStrategy):
    """Risk-On/Risk-Off (RORO) Baseline."""

    def __init__(
        self,
        vol_threshold: float = 0.25,
        ma_periods: Tuple[int, ...] = (20, 50, 200),
        momentum_periods: Tuple[int, ...] = (5, 20, 60),
    ):
        config = StrategyConfig(
            name="RORO",
            category="academic_baseline",
            parameters={"vol_threshold": vol_threshold},
            regime_hypothesis="Defensive drawdown protection",
            failure_mode="Misses rallies after corrections",
        )
        super().__init__(config)
        self.vol_threshold = vol_threshold
        self.ma_periods = ma_periods
        self.momentum_periods = momentum_periods

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        max_period = max(max(self.ma_periods), max(self.momentum_periods))
        if idx < max_period + 1:
            return StrategyResult(
                signal=Signal.CASH,
                confidence=0.0,
                raw_indicator=0.0,
                metadata={"reason": "Insufficient data"}
            )

        close = data["close"].iloc[:idx + 1]
        current_price = close.iloc[-1]

        risk_score = 0
        max_score = 0

        returns = close.pct_change().iloc[-20:]
        current_vol = returns.std() * np.sqrt(252)
        if current_vol > self.vol_threshold:
            risk_score += 2
        max_score += 2

        for period in self.ma_periods:
            ma = close.iloc[-period:].mean()
            if current_price < ma:
                risk_score += 1
            max_score += 1

        neg_momentum_count = 0
        for period in self.momentum_periods:
            ret = (current_price / close.iloc[-period]) - 1
            if ret < 0:
                neg_momentum_count += 1

        if neg_momentum_count >= 2:
            risk_score += 2
        max_score += 2

        risk_off_pct = risk_score / max_score

        if risk_off_pct > 0.6:
            signal = Signal.CASH
            confidence = risk_off_pct
        elif risk_off_pct > 0.4:
            signal = Signal.LONG
            confidence = 0.3
        else:
            signal = Signal.LONG
            confidence = 1.0 - risk_off_pct

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=risk_off_pct,
            metadata={"risk_score": risk_score, "risk_off_pct": risk_off_pct}
        )


ACADEMIC_BASELINES = [
    VolatilityTargetingBaseline,
    TimeSeriesMomentumBaseline,
    VolatilityRiskPremiumBaseline,
    TrendFollowingEnsembleBaseline,
    AdaptiveMomentumBaseline,
    RiskOnRiskOffBaseline,
]


def get_all_academic_baselines() -> list:
    """Return instances of all academic baseline strategies."""
    return [cls() for cls in ACADEMIC_BASELINES]

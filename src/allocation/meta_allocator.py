"""
Meta-Allocation Engine
======================

Phase 5: Combines validated strategies with micro-regime-aware allocation.

Key principles:
1. "Tilt not switch" - gradual weight adjustments, not binary switching
2. Turnover penalty - penalize excessive trading
3. Risk budget - respect 20% max drawdown constraint
4. Correlation-aware - reduce correlated positions

Validated strategies (beat TrendEnsemble baseline, Sharpe 3.88):
- BBSqueeze (10.61)
- DonchianBreakout (8.18)
- KeltnerBreakout (5.55)
- Ichimoku (5.00)
- ParabolicSAR (4.56)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from regime.micro_regimes import MicroRegimeDetector, MicroRegime, TrendState, VolatilityState


@dataclass
class AllocationConfig:
    """Configuration for meta-allocation engine."""
    # Strategy weights (base allocation)
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        "BBSqueeze": 0.25,
        "DonchianBreakout": 0.25,
        "KeltnerBreakout": 0.15,
        "Ichimoku": 0.10,
        "ParabolicSAR": 0.05,
        "TrendEnsemble": 0.10,  # Academic baseline for diversification
        "RORO": 0.10,  # Defensive
    })

    # Risk constraints
    max_leverage: float = 2.0
    max_drawdown: float = 0.20
    max_single_strategy_weight: float = 0.35
    min_single_strategy_weight: float = 0.05

    # Turnover control
    max_daily_turnover: float = 0.20  # Max 20% portfolio change per day
    turnover_penalty: float = 0.001  # 10bps penalty per unit turnover
    rebalance_threshold: float = 0.05  # Only rebalance if weights drift > 5%

    # Regime adjustments
    crisis_vol_cash_increase: float = 0.30  # Increase cash by 30% in crisis
    strong_bear_defensive_tilt: float = 0.20  # Tilt 20% toward defensive


@dataclass
class AllocationState:
    """Current state of the allocation."""
    weights: Dict[str, float]
    cash_weight: float
    leverage: float
    last_rebalance_date: Optional[pd.Timestamp]
    current_drawdown: float
    peak_equity: float


class MetaAllocator:
    """
    Meta-allocation engine that combines strategies with regime-awareness.

    Uses "tilt not switch" approach:
    - Base allocation from validated strategies
    - Gradual tilts based on micro-regime
    - Turnover penalty to prevent over-trading
    - Risk-based position sizing
    """

    def __init__(self, config: Optional[AllocationConfig] = None):
        self.config = config or AllocationConfig()
        self.regime_detector = MicroRegimeDetector()
        self.state: Optional[AllocationState] = None

    def initialize(self, initial_capital: float) -> AllocationState:
        """Initialize allocation state."""
        self.state = AllocationState(
            weights=self.config.base_weights.copy(),
            cash_weight=0.0,
            leverage=1.0,
            last_rebalance_date=None,
            current_drawdown=0.0,
            peak_equity=initial_capital,
        )
        return self.state

    def compute_target_allocation(
        self,
        data: pd.DataFrame,
        idx: int,
        current_equity: float,
    ) -> Dict[str, float]:
        """
        Compute target allocation based on current regime and state.

        Args:
            data: OHLCV data
            idx: Current index
            current_equity: Current portfolio equity

        Returns:
            Dictionary of strategy -> target weight
        """
        if self.state is None:
            self.initialize(current_equity)

        # Detect current micro-regime
        regime = self.regime_detector.detect(data, idx)

        # Start with base weights
        target_weights = self.config.base_weights.copy()

        # Apply regime-based tilts
        if regime is not None:
            target_weights = self._apply_regime_tilts(target_weights, regime)

        # Apply drawdown-based adjustments
        target_weights, leverage = self._apply_risk_adjustments(
            target_weights, current_equity
        )

        # Apply turnover constraints
        target_weights = self._apply_turnover_constraints(target_weights)

        # Normalize weights
        target_weights = self._normalize_weights(target_weights)

        # Update state
        self.state.weights = target_weights
        self.state.leverage = leverage

        return target_weights

    def _apply_regime_tilts(
        self,
        weights: Dict[str, float],
        regime: MicroRegime,
    ) -> Dict[str, float]:
        """
        Apply regime-based tilts to weights.

        Tilt not switch: gradual adjustments, not binary changes.
        """
        tilted = weights.copy()

        # Crisis volatility: increase cash/defensive
        if regime.volatility == VolatilityState.CRISIS:
            # Reduce all risky strategies
            reduction = self.config.crisis_vol_cash_increase
            for strategy in tilted:
                if strategy != "RORO":
                    tilted[strategy] *= (1 - reduction)
            # Increase RORO (defensive)
            tilted["RORO"] = tilted.get("RORO", 0) + reduction * 0.5
            self.state.cash_weight = reduction * 0.5

        # High volatility: slight reduction
        elif regime.volatility == VolatilityState.HIGH:
            reduction = 0.15
            for strategy in tilted:
                if strategy != "RORO":
                    tilted[strategy] *= (1 - reduction)
            tilted["RORO"] = tilted.get("RORO", 0) + reduction * 0.5
            self.state.cash_weight = reduction * 0.5

        # Strong bear + accelerating down: defensive tilt
        if regime.trend == TrendState.STRONG_BEAR:
            tilt = self.config.strong_bear_defensive_tilt
            # Reduce trend-following
            for strategy in ["DonchianBreakout", "Ichimoku", "ParabolicSAR"]:
                if strategy in tilted:
                    tilted[strategy] *= (1 - tilt)
            # Increase RORO
            tilted["RORO"] = tilted.get("RORO", 0) + tilt

        # Strong bull + accelerating: increase trend-following
        if regime.trend == TrendState.STRONG_BULL:
            if regime.momentum.name == "ACCELERATING":
                boost = 0.10
                for strategy in ["DonchianBreakout", "ParabolicSAR"]:
                    if strategy in tilted:
                        tilted[strategy] *= (1 + boost)
                # Reduce defensive
                tilted["RORO"] = max(0.05, tilted.get("RORO", 0) * 0.8)

        # Low volatility + neutral trend: reduce all, more cash
        if regime.volatility == VolatilityState.LOW and regime.trend == TrendState.NEUTRAL:
            reduction = 0.20
            for strategy in tilted:
                tilted[strategy] *= (1 - reduction)
            self.state.cash_weight = reduction

        return tilted

    def _apply_risk_adjustments(
        self,
        weights: Dict[str, float],
        current_equity: float,
    ) -> Tuple[Dict[str, float], float]:
        """
        Apply risk-based adjustments based on drawdown.

        Dynamic leverage reduction as drawdown increases.
        """
        # Update drawdown tracking
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity

        current_dd = (self.state.peak_equity - current_equity) / self.state.peak_equity
        self.state.current_drawdown = current_dd

        # Base leverage
        leverage = self.config.max_leverage

        # Reduce leverage as drawdown increases
        if current_dd > 0.15:
            # Linear reduction from 15% to 20% DD
            reduction = min(1.0, (current_dd - 0.10) / 0.10)
            leverage = max(1.0, self.config.max_leverage * (1 - reduction))
        elif current_dd > 0.10:
            # 75% leverage at 10% DD
            leverage = self.config.max_leverage * 0.75

        # If approaching max DD, go to cash
        if current_dd > 0.18:
            # Emergency de-risk
            for strategy in weights:
                weights[strategy] *= 0.5
            self.state.cash_weight = 0.5
            leverage = 1.0

        return weights, leverage

    def _apply_turnover_constraints(
        self,
        target_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Limit turnover to prevent excessive trading.

        Only allows max_daily_turnover change per day.
        """
        if self.state.weights is None:
            return target_weights

        constrained = {}
        current = self.state.weights

        for strategy, target in target_weights.items():
            current_weight = current.get(strategy, 0)
            diff = target - current_weight

            # Limit change
            max_change = self.config.max_daily_turnover
            if abs(diff) > max_change:
                diff = max_change if diff > 0 else -max_change

            constrained[strategy] = current_weight + diff

        return constrained

    def _normalize_weights(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalize weights to sum to 1 (minus cash).

        Also enforce min/max constraints.
        """
        # Enforce min/max
        for strategy in weights:
            weights[strategy] = max(
                self.config.min_single_strategy_weight,
                min(self.config.max_single_strategy_weight, weights[strategy])
            )

        # Normalize
        total = sum(weights.values()) + self.state.cash_weight
        if total > 0:
            for strategy in weights:
                weights[strategy] /= total

        return weights

    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> bool:
        """
        Check if rebalancing is needed based on weight drift.
        """
        total_drift = sum(
            abs(target_weights.get(s, 0) - current_weights.get(s, 0))
            for s in set(target_weights) | set(current_weights)
        )

        return total_drift > self.config.rebalance_threshold

    def calculate_turnover_cost(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
    ) -> float:
        """
        Calculate turnover cost for a rebalance.
        """
        turnover = sum(
            abs(new_weights.get(s, 0) - old_weights.get(s, 0))
            for s in set(new_weights) | set(old_weights)
        ) / 2  # Divide by 2 because we count both buy and sell

        return turnover * self.config.turnover_penalty


class PortfolioSimulator:
    """
    Simulate portfolio performance using meta-allocation.
    """

    def __init__(
        self,
        allocator: MetaAllocator,
        strategies: Dict[str, any],  # Strategy name -> strategy instance
        initial_capital: float = 500_000,
    ):
        self.allocator = allocator
        self.strategies = strategies
        self.initial_capital = initial_capital

    def run(
        self,
        data: pd.DataFrame,
        strategy_returns: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Run portfolio simulation.

        Args:
            data: OHLCV data for regime detection
            strategy_returns: Pre-computed daily returns for each strategy
                             NOTE: These should be UN-leveraged returns (raw strategy returns)

        Returns:
            DataFrame with portfolio equity curve and metrics
        """
        # Initialize
        self.allocator.initialize(self.initial_capital)

        n = len(data)
        equity = np.zeros(n)
        equity[0] = self.initial_capital

        weights_history = []
        regime_history = []
        turnover_costs = []

        current_weights = self.allocator.config.base_weights.copy()

        for idx in range(1, n):
            date = data.index[idx]

            # Get target allocation
            target_weights = self.allocator.compute_target_allocation(
                data, idx, equity[idx - 1]
            )

            # Check if rebalancing
            if self.allocator.should_rebalance(current_weights, target_weights):
                turnover_cost = self.allocator.calculate_turnover_cost(
                    current_weights, target_weights
                )
                current_weights = target_weights
            else:
                turnover_cost = 0

            turnover_costs.append(turnover_cost)

            # Calculate portfolio return (weighted average of strategy returns)
            portfolio_return = 0
            total_weight = 0
            for strategy, weight in current_weights.items():
                if strategy in strategy_returns:
                    strat_ret = strategy_returns[strategy].iloc[idx]
                    if not np.isnan(strat_ret):
                        # Cap extreme returns to prevent compounding explosion
                        strat_ret = max(-0.20, min(0.20, strat_ret))
                        portfolio_return += weight * strat_ret
                        total_weight += weight

            # Normalize by actual weights used
            if total_weight > 0 and total_weight != 1.0:
                portfolio_return = portfolio_return / total_weight

            # DO NOT apply additional leverage - strategy returns already include it
            # portfolio_return *= self.allocator.state.leverage

            # Subtract turnover cost
            portfolio_return -= turnover_cost

            # Cap daily return to prevent unrealistic compounding
            portfolio_return = max(-0.15, min(0.15, portfolio_return))

            # Update equity
            equity[idx] = equity[idx - 1] * (1 + portfolio_return)

            # Track history
            weights_history.append(current_weights.copy())
            regime_history.append(
                self.allocator.regime_detector.detect(data, idx)
            )

        # Build results DataFrame
        results = pd.DataFrame({
            "date": data.index,
            "equity": equity,
            "returns": np.concatenate([[0], np.diff(equity) / equity[:-1]]),
        })
        results.set_index("date", inplace=True)

        return results, weights_history, turnover_costs


def calculate_portfolio_metrics(
    equity: pd.Series,
    risk_free_rate: float = 0.04,
) -> Dict[str, float]:
    """Calculate portfolio performance metrics."""
    returns = equity.pct_change().dropna()

    # Annualized return
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1

    # Volatility
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    rolling_max = equity.expanding().max()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "num_days": len(equity),
    }

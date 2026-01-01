#!/usr/bin/env python3
"""
Phase 5: Meta-Allocation Backtest
=================================

Tests the regime-aware meta-allocation engine against:
1. Equal-weight portfolio of validated strategies
2. Best individual strategy (BBSqueeze)
3. Buy & Hold baseline

Uses Phase 4 validated strategies:
- BBSqueeze (OOS Sharpe 16.28)
- DonchianBreakout (OOS Sharpe 13.06)
- KeltnerBreakout (OOS Sharpe 8.46)
- ParabolicSAR (OOS Sharpe 8.04)
- ATRBreakout (OOS Sharpe 7.63)

Usage:
    python scripts/phase5_meta_allocation.py
"""

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from data.loader import QQQDataLoader
from backtesting.engine import BacktestEngine, BacktestConfig
from regime.micro_regimes import MicroRegimeDetector, TrendState, VolatilityState

# Import validated strategies
from strategies import (
    BBSqueezeStrategy,
    DonchianBreakoutStrategy,
    KeltnerBreakoutStrategy,
    ParabolicSARStrategy,
    ATRBreakoutStrategy,
    IchimokuStrategy,
    BuyAndHoldStrategy,
    SMA200Strategy,
)


# Validated strategies from Phase 4
VALIDATED_STRATEGIES = {
    "BBSqueeze": BBSqueezeStrategy,
    "DonchianBreakout": DonchianBreakoutStrategy,
    "KeltnerBreakout": KeltnerBreakoutStrategy,
    "ParabolicSAR": ParabolicSARStrategy,
    "ATRBreakout": ATRBreakoutStrategy,
    "Ichimoku": IchimokuStrategy,
}


@dataclass
class MetaAllocationConfig:
    """Configuration for meta-allocation."""
    # Base weights from Phase 4 OOS Sharpe ratios
    base_weights: Dict[str, float] = field(default_factory=lambda: {
        "BBSqueeze": 0.30,       # Highest OOS Sharpe
        "DonchianBreakout": 0.25,
        "KeltnerBreakout": 0.15,
        "ParabolicSAR": 0.15,
        "ATRBreakout": 0.10,
        "Ichimoku": 0.05,
    })

    # Turnover constraints - TIGHTENED to meet 50% annual target
    max_daily_turnover: float = 0.05  # Reduced from 0.20
    rebalance_threshold: float = 0.10  # Increased from 0.05 (only rebalance if drift > 10%)
    rebalance_cooldown_days: int = 10  # Minimum days between rebalances
    turnover_cost_bps: float = 10  # 10 bps per unit turnover

    # Risk constraints
    max_drawdown: float = 0.25
    emergency_cash_threshold: float = 0.20  # Go to 50% cash at 20% DD


class SimpleMetaAllocator:
    """
    Simplified meta-allocator using validated strategies.

    Implements:
    1. Base weights from Phase 4 OOS performance
    2. Regime-based tilts (tilt not switch)
    3. Turnover constraints
    4. Drawdown-based deleveraging
    """

    def __init__(self, config: Optional[MetaAllocationConfig] = None):
        self.config = config or MetaAllocationConfig()
        self.regime_detector = MicroRegimeDetector()
        self.current_weights = self.config.base_weights.copy()
        self.peak_equity = None
        self.current_drawdown = 0.0
        self.last_rebalance_idx = 0  # Track last rebalance for cooldown
        self.last_regime = None  # Track regime for change detection

    def compute_target_weights(
        self,
        data: pd.DataFrame,
        idx: int,
        current_equity: float,
    ) -> Dict[str, float]:
        """Compute target allocation weights."""
        # Initialize peak equity
        if self.peak_equity is None:
            self.peak_equity = current_equity

        # Update drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity

        # Detect regime
        regime = self.regime_detector.detect(data, idx)

        # Check if we should rebalance (cooldown + regime change + threshold)
        days_since_rebalance = idx - self.last_rebalance_idx
        regime_changed = (self.last_regime is not None and regime is not None and
                         (regime.trend != self.last_regime.trend or
                          regime.volatility != self.last_regime.volatility))

        should_rebalance = (
            days_since_rebalance >= self.config.rebalance_cooldown_days and
            regime_changed
        )

        if not should_rebalance:
            # No rebalance - keep current weights
            return self.current_weights

        # Start with base weights
        target = self.config.base_weights.copy()

        if regime is not None:
            # Apply regime tilts
            target = self._apply_regime_tilts(target, regime)
            self.last_regime = regime

        # Apply drawdown adjustments
        target = self._apply_drawdown_adjustments(target)

        # Apply turnover constraints
        target = self._apply_turnover_constraints(target)

        # Update tracking
        self.current_weights = target
        self.last_rebalance_idx = idx

        return target

    def _apply_regime_tilts(
        self,
        weights: Dict[str, float],
        regime,
    ) -> Dict[str, float]:
        """Apply regime-based tilts."""
        tilted = weights.copy()

        # High/Crisis volatility: reduce aggressive strategies, favor volatility strategies
        if regime.volatility in [VolatilityState.HIGH, VolatilityState.CRISIS]:
            reduction = 0.20 if regime.volatility == VolatilityState.HIGH else 0.35

            # Reduce trend-following (they struggle in choppy markets)
            for strat in ["ParabolicSAR", "Ichimoku"]:
                if strat in tilted:
                    tilted[strat] *= (1 - reduction)

            # Increase breakout strategies (they benefit from vol)
            for strat in ["BBSqueeze", "KeltnerBreakout", "ATRBreakout"]:
                if strat in tilted:
                    tilted[strat] *= (1 + reduction * 0.5)

        # Strong bear: favor Donchian (958% in bear from Phase 4)
        if regime.trend == TrendState.STRONG_BEAR:
            tilted["DonchianBreakout"] = tilted.get("DonchianBreakout", 0) * 1.5

        # Low volatility: reduce exposure (markets may be topping)
        if regime.volatility == VolatilityState.LOW:
            for strat in tilted:
                tilted[strat] *= 0.85

        # Normalize
        total = sum(tilted.values())
        if total > 0:
            tilted = {k: v / total for k, v in tilted.items()}

        return tilted

    def _apply_drawdown_adjustments(
        self,
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Reduce exposure as drawdown increases."""
        if self.current_drawdown < 0.10:
            return weights

        # Linear reduction from 10% to 25% DD
        reduction = min(0.5, (self.current_drawdown - 0.10) / 0.15)

        adjusted = {k: v * (1 - reduction) for k, v in weights.items()}

        return adjusted

    def _apply_turnover_constraints(
        self,
        target: Dict[str, float],
    ) -> Dict[str, float]:
        """Limit daily turnover."""
        constrained = {}

        for strat, target_weight in target.items():
            current_weight = self.current_weights.get(strat, 0)
            diff = target_weight - current_weight

            # Limit change
            max_change = self.config.max_daily_turnover
            if abs(diff) > max_change:
                diff = max_change if diff > 0 else -max_change

            constrained[strat] = current_weight + diff

        return constrained


def run_strategy_backtest(
    strategy_class,
    data: pd.DataFrame,
    config: BacktestConfig,
) -> pd.Series:
    """Run backtest for a single strategy and return daily returns."""
    engine = BacktestEngine(config=config)
    strategy = strategy_class()
    result = engine.run(strategy, data)
    return result.returns


def run_meta_allocation_backtest(
    data: pd.DataFrame,
    strategy_returns: Dict[str, pd.Series],
    config: MetaAllocationConfig,
    initial_capital: float = 500_000,
) -> pd.DataFrame:
    """Run meta-allocation backtest."""
    allocator = SimpleMetaAllocator(config)

    n = len(data)
    equity = np.zeros(n)
    equity[0] = initial_capital

    returns = np.zeros(n)
    weights_history = []

    for idx in range(1, n):
        # Get target weights
        target_weights = allocator.compute_target_weights(
            data, idx, equity[idx - 1]
        )

        # Calculate portfolio return
        portfolio_return = 0
        total_weight = 0

        for strat, weight in target_weights.items():
            if strat in strategy_returns:
                strat_ret = strategy_returns[strat].iloc[idx]
                if not pd.isna(strat_ret):
                    # Cap extreme returns
                    strat_ret = max(-0.15, min(0.15, strat_ret))
                    portfolio_return += weight * strat_ret
                    total_weight += weight

        # Normalize
        if total_weight > 0 and total_weight != 1.0:
            portfolio_return = portfolio_return / total_weight

        # Calculate turnover cost
        old_weights = allocator.current_weights if idx > 1 else allocator.config.base_weights
        turnover = sum(abs(target_weights.get(s, 0) - old_weights.get(s, 0)) for s in set(target_weights) | set(old_weights)) / 2
        turnover_cost = turnover * config.turnover_cost_bps / 10000

        portfolio_return -= turnover_cost

        # Update equity
        equity[idx] = equity[idx - 1] * (1 + portfolio_return)
        returns[idx] = portfolio_return
        weights_history.append(target_weights.copy())

    results = pd.DataFrame({
        "date": data.index,
        "equity": equity,
        "returns": returns,
    })
    results.set_index("date", inplace=True)

    return results, weights_history


def run_equal_weight_backtest(
    strategy_returns: Dict[str, pd.Series],
    initial_capital: float = 500_000,
) -> pd.DataFrame:
    """Run equal-weight portfolio backtest."""
    n_strategies = len(strategy_returns)
    weight = 1.0 / n_strategies

    # Get common index
    first_key = list(strategy_returns.keys())[0]
    n = len(strategy_returns[first_key])

    equity = np.zeros(n)
    equity[0] = initial_capital
    returns = np.zeros(n)

    for idx in range(1, n):
        portfolio_return = 0
        count = 0

        for strat_ret in strategy_returns.values():
            ret = strat_ret.iloc[idx]
            if not pd.isna(ret):
                ret = max(-0.15, min(0.15, ret))
                portfolio_return += weight * ret
                count += 1

        if count > 0 and count != n_strategies:
            portfolio_return = portfolio_return * n_strategies / count

        equity[idx] = equity[idx - 1] * (1 + portfolio_return)
        returns[idx] = portfolio_return

    # Use same index as strategy returns
    index = strategy_returns[first_key].index

    results = pd.DataFrame({
        "equity": equity,
        "returns": returns,
    }, index=index)

    return results


def calculate_metrics(equity: pd.Series, risk_free_rate: float = 0.04) -> Dict:
    """Calculate performance metrics."""
    returns = equity.pct_change().dropna()

    # Total and annualized return
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1

    # Volatility
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    rolling_max = equity.expanding().max()
    drawdowns = (equity - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
    }


def main():
    print("=" * 80)
    print("CHEN STRATEGY - PHASE 5: META-ALLOCATION BACKTEST")
    print("=" * 80)

    # Load data
    print("\nLoading QQQ data...")
    loader = QQQDataLoader(start_date="2000-01-01", end_date="2024-12-31")
    data = loader.fetch()
    data = loader.add_returns()
    print(f"  Loaded {len(data)} trading days")

    # Backtest config
    config = BacktestConfig(
        initial_capital=500_000,
        max_leverage=3.0,
        max_drawdown=0.25,
        warmup_period=252,
    )

    # Run individual strategy backtests
    print("\n" + "=" * 80)
    print("RUNNING INDIVIDUAL STRATEGY BACKTESTS")
    print("=" * 80)

    strategy_returns = {}
    for name, strategy_class in VALIDATED_STRATEGIES.items():
        print(f"  Backtesting {name}...")
        returns = run_strategy_backtest(strategy_class, data, config)
        strategy_returns[name] = returns

    # Also get BuyAndHold for comparison
    print("  Backtesting BuyAndHold...")
    bh_returns = run_strategy_backtest(BuyAndHoldStrategy, data, config)

    # Run meta-allocation backtest
    print("\n" + "=" * 80)
    print("RUNNING META-ALLOCATION BACKTEST")
    print("=" * 80)

    meta_config = MetaAllocationConfig()
    meta_results, weights_history = run_meta_allocation_backtest(
        data, strategy_returns, meta_config
    )

    # Run equal-weight backtest
    print("\nRunning equal-weight portfolio backtest...")
    eq_results = run_equal_weight_backtest(strategy_returns)

    # Run best individual (BBSqueeze) for comparison
    print("\nRunning BBSqueeze (best individual) backtest...")
    bb_engine = BacktestEngine(config=config)
    bb_result = bb_engine.run(BBSqueezeStrategy(), data)

    # Calculate metrics for all
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    results = {
        "Meta-Allocation": calculate_metrics(meta_results["equity"]),
        "Equal-Weight": calculate_metrics(eq_results["equity"]),
        "BBSqueeze (Best)": {
            "total_return": bb_result.metrics.total_return,
            "annualized_return": bb_result.metrics.annualized_return,
            "annualized_volatility": bb_result.metrics.annualized_volatility,
            "sharpe_ratio": bb_result.metrics.sharpe_ratio,
            "sortino_ratio": bb_result.metrics.sortino_ratio,
            "calmar_ratio": bb_result.metrics.calmar_ratio,
            "max_drawdown": bb_result.metrics.max_drawdown,
        },
    }

    # Print comparison table
    print("\n" + "-" * 80)
    print(f"{'Strategy':<25} {'Sharpe':>10} {'Ann. Ret':>12} {'Max DD':>10} {'Calmar':>10}")
    print("-" * 80)

    for name, metrics in results.items():
        sharpe = metrics["sharpe_ratio"]
        ann_ret = metrics["annualized_return"] * 100
        max_dd = metrics["max_drawdown"] * 100
        calmar = metrics["calmar_ratio"]
        print(f"{name:<25} {sharpe:>10.2f} {ann_ret:>11.1f}% {max_dd:>9.1f}% {calmar:>10.2f}")

    print("-" * 80)

    # Phase 5 pass criteria
    print("\n" + "=" * 80)
    print("PHASE 5 CRITERIA CHECK")
    print("=" * 80)

    meta_sharpe = results["Meta-Allocation"]["sharpe_ratio"]
    best_sharpe = results["BBSqueeze (Best)"]["sharpe_ratio"]
    meta_dd = abs(results["Meta-Allocation"]["max_drawdown"])

    # Calculate approximate turnover
    avg_turnover = 0
    if len(weights_history) > 1:
        for i in range(1, len(weights_history)):
            turnover = sum(abs(weights_history[i].get(s, 0) - weights_history[i-1].get(s, 0))
                          for s in set(weights_history[i]) | set(weights_history[i-1])) / 2
            avg_turnover += turnover
        avg_turnover = avg_turnover / (len(weights_history) - 1) * 252  # Annualized

    criteria = [
        ("Meta-allocation Sharpe > 1.0", meta_sharpe > 1.0, f"{meta_sharpe:.2f}"),
        ("Max drawdown < 25%", meta_dd < 0.25, f"{meta_dd*100:.1f}%"),
        ("Turnover < 50% annually", avg_turnover < 0.50, f"{avg_turnover*100:.1f}%"),
    ]

    print()
    all_pass = True
    for criterion, passed, actual in criteria:
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  {criterion}: {status} (actual: {actual})")

    print()
    print(f"PHASE 5 OVERALL: {'PASS' if all_pass else 'NEEDS REVIEW'}")

    # Save results
    output_dir = Path("results/phase5")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save equity curves
    meta_results.to_csv(output_dir / f"meta_allocation_equity_{timestamp}.csv")
    eq_results.to_csv(output_dir / f"equal_weight_equity_{timestamp}.csv")

    # Save comparison
    comparison_df = pd.DataFrame(results).T
    comparison_df.to_csv(output_dir / f"phase5_comparison_{timestamp}.csv")

    print(f"\n\nResults saved to: {output_dir}")

    print("\n" + "=" * 80)
    print("PHASE 5 BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 5: Meta-Allocation Portfolio Backtest
============================================

Runs the complete portfolio simulation using:
- 5 validated strategies + 2 academic baselines
- Micro-regime-aware allocation
- Tilt-not-switch approach
- Turnover penalties
- Risk-based position sizing

IMPORTANT: Uses 1x leverage at strategy level, applies portfolio leverage separately.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data.loader import QQQDataLoader
from backtesting.engine import BacktestEngine, BacktestConfig
from regime.micro_regimes import MicroRegimeDetector, TrendState, VolatilityState


def run_unleveraged_backtests(data: pd.DataFrame) -> dict:
    """Run backtests for each strategy at 1x leverage to get base returns."""
    from strategies.trend_following import (
        DonchianBreakoutStrategy, IchimokuStrategy, ParabolicSARStrategy
    )
    from strategies.volatility import BBSqueezeStrategy, KeltnerBreakoutStrategy
    from strategies.academic_baselines import (
        TrendFollowingEnsembleBaseline, RiskOnRiskOffBaseline
    )

    strategies = {
        "BBSqueeze": BBSqueezeStrategy(),
        "DonchianBreakout": DonchianBreakoutStrategy(),
        "KeltnerBreakout": KeltnerBreakoutStrategy(),
        "Ichimoku": IchimokuStrategy(),
        "ParabolicSAR": ParabolicSARStrategy(),
        "TrendEnsemble": TrendFollowingEnsembleBaseline(),
        "RORO": RiskOnRiskOffBaseline(),
    }

    # Use 1x leverage to get base returns
    config = BacktestConfig(
        initial_capital=500_000,
        max_leverage=1.0,  # No leverage at strategy level
        max_drawdown=0.25,  # Relaxed for unleveraged
        warmup_period=200,
    )
    engine = BacktestEngine(config=config)

    strategy_returns = {}

    print("Running unleveraged strategy backtests...")
    for name, strategy in strategies.items():
        print(f"  {name}...", end=" ", flush=True)
        try:
            result = engine.run(strategy, data)
            strategy_returns[name] = result.returns
            sharpe = result.metrics.sharpe_ratio
            print(f"Sharpe={sharpe:.2f} (1x leverage)")
        except Exception as e:
            print(f"ERROR: {e}")
            strategy_returns[name] = pd.Series(0, index=data.index)

    return strategy_returns


def run_portfolio_simulation(
    data: pd.DataFrame,
    strategy_returns: dict,
    base_weights: dict,
    max_leverage: float = 2.0,
    initial_capital: float = 500_000,
) -> tuple:
    """
    Run portfolio simulation with regime-aware allocation.

    Returns: (equity_series, metrics_dict)
    """
    regime_detector = MicroRegimeDetector()

    n = len(data)
    equity = np.zeros(n)
    equity[0] = initial_capital
    peak_equity = initial_capital

    daily_returns = []
    regime_history = []

    for idx in range(1, n):
        # Detect regime
        regime = regime_detector.detect(data, idx)

        # Calculate current drawdown
        if equity[idx - 1] > peak_equity:
            peak_equity = equity[idx - 1]
        current_dd = (peak_equity - equity[idx - 1]) / peak_equity

        # Adjust weights based on regime and drawdown
        weights = base_weights.copy()
        leverage = max_leverage

        if regime is not None:
            # Crisis volatility: reduce exposure
            if regime.volatility == VolatilityState.CRISIS:
                leverage *= 0.5
                weights["RORO"] = weights.get("RORO", 0) * 1.5

            # High volatility: slight reduction
            elif regime.volatility == VolatilityState.HIGH:
                leverage *= 0.75

            # Strong bear: defensive tilt
            if regime.trend == TrendState.STRONG_BEAR:
                leverage *= 0.7
                weights["RORO"] = weights.get("RORO", 0) * 1.3

        # Drawdown-based leverage reduction
        if current_dd > 0.15:
            leverage *= 0.5
        elif current_dd > 0.10:
            leverage *= 0.75

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate portfolio return
        portfolio_return = 0
        for strategy, weight in weights.items():
            if strategy in strategy_returns:
                strat_ret = strategy_returns[strategy].iloc[idx]
                if not np.isnan(strat_ret):
                    portfolio_return += weight * strat_ret

        # Apply leverage
        portfolio_return *= leverage

        # Cap extreme returns
        portfolio_return = max(-0.10, min(0.10, portfolio_return))

        daily_returns.append(portfolio_return)

        # Update equity
        equity[idx] = equity[idx - 1] * (1 + portfolio_return)

        regime_history.append(regime.code if regime else "N/A")

    # Calculate metrics
    returns_series = pd.Series(daily_returns, index=data.index[1:])
    equity_series = pd.Series(equity, index=data.index)

    # Annualized metrics
    total_return = equity[-1] / equity[0] - 1
    years = len(data) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    metrics = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "final_equity": equity[-1],
    }

    return equity_series, metrics, returns_series


def main():
    print()
    print("=" * 70)
    print("  PHASE 5: META-ALLOCATION PORTFOLIO BACKTEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    print()

    # Load data
    print("Loading QQQ data (2000-2024)...")
    loader = QQQDataLoader(start_date="2000-01-01", end_date="2024-12-31")
    data = loader.fetch()

    if data is None:
        print("ERROR: Could not load data")
        return

    print(f"Loaded {len(data)} trading days")
    print()

    # Run unleveraged backtests
    strategy_returns = run_unleveraged_backtests(data)

    # Base weights
    base_weights = {
        "BBSqueeze": 0.25,
        "DonchianBreakout": 0.25,
        "KeltnerBreakout": 0.15,
        "Ichimoku": 0.10,
        "ParabolicSAR": 0.05,
        "TrendEnsemble": 0.10,
        "RORO": 0.10,
    }

    print()
    print("=" * 70)
    print("  PORTFOLIO SIMULATION")
    print("=" * 70)
    print()

    print("Base allocation:")
    for strategy, weight in base_weights.items():
        print(f"  {strategy}: {weight*100:.0f}%")
    print()
    print("Max leverage: 2.0x (dynamically adjusted by regime)")
    print()

    # Run portfolio simulation
    print("Running portfolio simulation...")
    equity, metrics, returns = run_portfolio_simulation(
        data, strategy_returns, base_weights, max_leverage=2.0
    )

    print()
    print("=" * 70)
    print("  PORTFOLIO RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'Initial Capital':<30} ${500000:>14,}")
    print(f"{'Final Equity':<30} ${metrics['final_equity']:>14,.0f}")
    print(f"{'Total Return':<30} {metrics['total_return']*100:>14.1f}%")
    print(f"{'Annualized Return':<30} {metrics['annualized_return']*100:>14.1f}%")
    print(f"{'Annualized Volatility':<30} {metrics['annualized_volatility']*100:>14.1f}%")
    print(f"{'Sharpe Ratio':<30} {metrics['sharpe_ratio']:>15.2f}")
    print(f"{'Max Drawdown':<30} {metrics['max_drawdown']*100:>14.1f}%")

    # Compare to benchmarks
    print()
    print("=" * 70)
    print("  COMPARISON TO BENCHMARKS")
    print("=" * 70)
    print()

    # Buy and hold (2x leveraged for fair comparison)
    bh_returns = data["close"].pct_change() * 2.0  # 2x leverage
    bh_returns = bh_returns.clip(-0.10, 0.10)  # Cap extreme
    bh_equity = 500_000 * (1 + bh_returns).cumprod()
    bh_equity.iloc[0] = 500_000

    bh_total_ret = bh_equity.iloc[-1] / bh_equity.iloc[0] - 1
    bh_years = len(data) / 252
    bh_ann_ret = (1 + bh_total_ret) ** (1 / bh_years) - 1
    bh_vol = bh_returns.std() * np.sqrt(252)
    bh_sharpe = (bh_ann_ret - 0.04) / bh_vol if bh_vol > 0 else 0
    bh_rolling_max = bh_equity.expanding().max()
    bh_max_dd = ((bh_equity - bh_rolling_max) / bh_rolling_max).min()

    # Equal weight portfolio (naive)
    equal_weights = {k: 1/7 for k in base_weights}
    eq_equity, eq_metrics, _ = run_portfolio_simulation(
        data, strategy_returns, equal_weights, max_leverage=2.0
    )

    print(f"{'Benchmark':<30} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10}")
    print("-" * 62)
    print(f"{'Meta-Allocation (regime-aware)':<30} {metrics['sharpe_ratio']:>8.2f} "
          f"{metrics['annualized_return']*100:>9.1f}% "
          f"{metrics['max_drawdown']*100:>9.1f}%")
    print(f"{'Equal-Weight Portfolio':<30} {eq_metrics['sharpe_ratio']:>8.2f} "
          f"{eq_metrics['annualized_return']*100:>9.1f}% "
          f"{eq_metrics['max_drawdown']*100:>9.1f}%")
    print(f"{'Buy & Hold QQQ (2x)':<30} {bh_sharpe:>8.2f} "
          f"{bh_ann_ret*100:>9.1f}% "
          f"{bh_max_dd*100:>9.1f}%")

    # Individual strategies (2x leveraged)
    print()
    print("Individual Strategies (2x leveraged):")
    config_2x = BacktestConfig(
        initial_capital=500_000,
        max_leverage=2.0,
        max_drawdown=0.20,
        warmup_period=200,
    )
    engine_2x = BacktestEngine(config=config_2x)

    from strategies.volatility import BBSqueezeStrategy
    from strategies.trend_following import DonchianBreakoutStrategy
    from strategies.academic_baselines import TrendFollowingEnsembleBaseline

    for name, strat_cls in [
        ("BBSqueeze", BBSqueezeStrategy),
        ("DonchianBreakout", DonchianBreakoutStrategy),
        ("TrendEnsemble", TrendFollowingEnsembleBaseline),
    ]:
        result = engine_2x.run(strat_cls(), data)
        print(f"  {name:<25} Sharpe={result.metrics.sharpe_ratio:>6.2f} "
              f"Return={result.metrics.annualized_return*100:>6.1f}% "
              f"MaxDD={result.metrics.max_drawdown*100:>6.1f}%")

    # Gate criteria check
    print()
    print("=" * 70)
    print("  PHASE 5 GATE CRITERIA CHECK")
    print("=" * 70)
    print()

    gates = [
        ("Sharpe > Buy&Hold (2x)", metrics['sharpe_ratio'] > bh_sharpe),
        ("Sharpe > Equal-Weight", metrics['sharpe_ratio'] > eq_metrics['sharpe_ratio']),
        ("Max DD < 20%", abs(metrics['max_drawdown']) < 0.20),
        ("Annualized Return > 15%", metrics['annualized_return'] > 0.15),
    ]

    all_passed = True
    for criterion, passed in gates:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {criterion}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 PHASE 5 GATE: PASSED")
    else:
        print("⚠️ PHASE 5 GATE: NEEDS REVIEW")

    # Save results
    os.makedirs("results", exist_ok=True)

    results_df = pd.DataFrame({
        "equity": equity,
        "returns": pd.concat([pd.Series([0]), returns]),
    }, index=data.index)
    results_df.to_csv("results/phase5_portfolio_equity.csv")
    print()
    print("Results saved to results/phase5_portfolio_equity.csv")

    # Summary
    summary = {
        "date": datetime.now().isoformat(),
        "portfolio_sharpe": metrics['sharpe_ratio'],
        "portfolio_return": metrics['annualized_return'],
        "portfolio_max_dd": metrics['max_drawdown'],
        "bh_sharpe": bh_sharpe,
        "equal_weight_sharpe": eq_metrics['sharpe_ratio'],
        "gate_passed": all_passed,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("results/phase5_summary.csv", index=False)
    print("Summary saved to results/phase5_summary.csv")

    return metrics


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Walk-Forward Validation Script
==============================

Runs walk-forward validation on strategies to detect overfitting.
Uses 3-year in-sample, 1-year out-of-sample windows.
"""

import sys
import os
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np

from data.loader import QQQDataLoader
from backtesting.engine import BacktestEngine, BacktestConfig, WalkForwardValidator
from strategies.base import BuyAndHoldStrategy, SMA200Strategy, GoldenCrossBaselineStrategy
from strategies.trend_following import (
    GoldenCrossStrategy, MACDTrendStrategy, ADXBreakoutStrategy,
    IchimokuStrategy, ParabolicSARStrategy, DonchianBreakoutStrategy,
)
from strategies.volatility import (
    ATRBreakoutStrategy, KeltnerBreakoutStrategy,
    VolTargetingStrategy, BBSqueezeStrategy,
)

# Top performing strategies to validate
TOP_STRATEGIES = {
    "BBSqueeze": BBSqueezeStrategy,
    "DonchianBreakout": DonchianBreakoutStrategy,
    "ParabolicSAR": ParabolicSARStrategy,
    "KeltnerBreakout": KeltnerBreakoutStrategy,
    "Ichimoku": IchimokuStrategy,
    "GoldenCross": GoldenCrossStrategy,
    "MACDTrend": MACDTrendStrategy,
}

BASELINES = {
    "BuyAndHold": BuyAndHoldStrategy,
    "SMA200": SMA200Strategy,
}


def run_walk_forward(strategy_name: str, strategy_class, data: pd.DataFrame, config: BacktestConfig):
    """Run walk-forward validation for a single strategy."""
    strategy = strategy_class()
    engine = BacktestEngine(config=config)

    # Use 2-year IS, 1-year OOS for 10-year dataset
    validator = WalkForwardValidator(
        is_window=252 * 2,  # 2 years in-sample
        oos_window=252,      # 1 year out-of-sample
        step=252,            # 1 year step
    )

    results = validator.validate(strategy, data, engine)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument("--strategy", type=str, help="Specific strategy to validate")
    parser.add_argument("--all", action="store_true", help="Validate all top strategies")
    parser.add_argument("--baselines", action="store_true", help="Include baselines")
    parser.add_argument("--start-date", type=str, default="2015-01-01", help="Start date")
    args = parser.parse_args()

    # Determine strategies to validate
    strategies_to_run = {}
    if args.strategy:
        if args.strategy in TOP_STRATEGIES:
            strategies_to_run[args.strategy] = TOP_STRATEGIES[args.strategy]
        elif args.strategy in BASELINES:
            strategies_to_run[args.strategy] = BASELINES[args.strategy]
        else:
            print(f"Unknown strategy: {args.strategy}")
            return
    elif args.all:
        strategies_to_run.update(TOP_STRATEGIES)
        if args.baselines:
            strategies_to_run.update(BASELINES)
    else:
        # Default: top 5 + baselines
        strategies_to_run = {
            "BBSqueeze": BBSqueezeStrategy,
            "DonchianBreakout": DonchianBreakoutStrategy,
            "KeltnerBreakout": KeltnerBreakoutStrategy,
            "SMA200": SMA200Strategy,
        }

    print(f"Running walk-forward validation on {len(strategies_to_run)} strategies...")
    print()

    # Load data
    print("Loading QQQ data...")
    loader = QQQDataLoader(start_date=args.start_date)
    data = loader.fetch()
    loader.add_returns()
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Total rows: {len(data)}")
    print()

    # Config
    config = BacktestConfig(
        initial_capital=500_000,
        max_leverage=3.0,
        max_drawdown=0.25,
        warmup_period=200,
    )

    # Run validation
    all_results = {}
    for name, strategy_class in strategies_to_run.items():
        print(f"Validating {name}...")
        try:
            results = run_walk_forward(name, strategy_class, data, config)
            all_results[name] = results

            # Print summary
            print(f"  Folds: {len(results['folds'])}")
            print(f"  Avg IS Sharpe:  {results['avg_is_sharpe']:.2f}")
            print(f"  Avg OOS Sharpe: {results['avg_oos_sharpe']:.2f}")
            print(f"  Sharpe Decay:   {results['avg_sharpe_decay']:.2f}")
            print(f"  Overfit Ratio:  {results['overfit_ratio']:.2f}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()

    # Summary table
    print("=" * 80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Strategy':<20} | {'IS Sharpe':>10} | {'OOS Sharpe':>10} | {'Decay':>8} | {'Overfit':>8} | {'Status':<12}")
    print("-" * 80)

    for name, results in sorted(all_results.items(), key=lambda x: x[1]['avg_oos_sharpe'], reverse=True):
        is_sharpe = results['avg_is_sharpe']
        oos_sharpe = results['avg_oos_sharpe']
        decay = results['avg_sharpe_decay']
        overfit = results['overfit_ratio']

        # Determine status
        if np.isnan(oos_sharpe) or np.isinf(overfit):
            status = "⚠️ INVALID"
        elif overfit > 3.0:
            status = "❌ OVERFIT"
        elif overfit > 2.0:
            status = "⚠️ SUSPECT"
        elif oos_sharpe > 0.5:
            status = "✅ ROBUST"
        else:
            status = "⚠️ WEAK"

        print(f"{name:<20} | {is_sharpe:>10.2f} | {oos_sharpe:>10.2f} | {decay:>8.2f} | {overfit:>8.2f} | {status:<12}")

    print()
    print("Legend:")
    print("  IS Sharpe: Average in-sample Sharpe ratio")
    print("  OOS Sharpe: Average out-of-sample Sharpe ratio (KEY METRIC)")
    print("  Decay: IS Sharpe - OOS Sharpe (lower is better)")
    print("  Overfit: IS/OOS ratio (closer to 1.0 is better, >3.0 is suspect)")
    print()
    print("Status:")
    print("  ✅ ROBUST: OOS Sharpe > 0.5, Overfit < 2.0")
    print("  ⚠️ SUSPECT: Overfit 2.0-3.0")
    print("  ❌ OVERFIT: Overfit > 3.0")
    print("  ⚠️ WEAK: OOS Sharpe < 0.5")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/walk_forward_validation_{timestamp}.csv"

    summary_rows = []
    for name, results in all_results.items():
        for _, fold in results['folds'].iterrows():
            row = fold.to_dict()
            row['strategy'] = name
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()

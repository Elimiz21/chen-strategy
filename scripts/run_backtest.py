#!/usr/bin/env python3
"""
Backtest Runner Script
======================

Run backtests on all strategies and generate performance reports.

Usage:
    python scripts/run_backtest.py --all
    python scripts/run_backtest.py --baselines
    python scripts/run_backtest.py --strategy GoldenCross
    python scripts/run_backtest.py --category trend_following
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from data.loader import QQQDataLoader
from data.versioning import DataVersioner
from backtesting.engine import BacktestEngine, BacktestConfig, WalkForwardValidator
from backtesting.cost_model import CostModel
from regime.detector import RulesBasedDetector

# Import all strategies
from strategies import (
    # Baselines
    BuyAndHoldStrategy,
    SMA200Strategy,
    GoldenCrossBaselineStrategy,
    # Trend-following
    GoldenCrossStrategy,
    MACDTrendStrategy,
    ADXBreakoutStrategy,
    IchimokuStrategy,
    ParabolicSARStrategy,
    DonchianBreakoutStrategy,
    # Mean-reversion
    RSIReversalStrategy,
    BollingerBounceStrategy,
    StochasticStrategy,
    WilliamsRStrategy,
    CCIReversalStrategy,
    # Volatility
    ATRBreakoutStrategy,
    KeltnerBreakoutStrategy,
    VolTargetingStrategy,
    BBSqueezeStrategy,
    # Volume
    OBVConfirmationStrategy,
    MFIReversalStrategy,
    VWAPReversionStrategy,
    # Momentum
    MomentumStrategy,
    AroonTrendStrategy,
    TRIXTrendStrategy,
)


# Strategy registry
STRATEGIES = {
    # Baselines
    "BuyAndHold": BuyAndHoldStrategy,
    "SMA200": SMA200Strategy,
    "GoldenCrossBaseline": GoldenCrossBaselineStrategy,
    # Trend-following
    "GoldenCross": GoldenCrossStrategy,
    "MACDTrend": MACDTrendStrategy,
    "ADXBreakout": ADXBreakoutStrategy,
    "Ichimoku": IchimokuStrategy,
    "ParabolicSAR": ParabolicSARStrategy,
    "DonchianBreakout": DonchianBreakoutStrategy,
    # Mean-reversion
    "RSIReversal": RSIReversalStrategy,
    "BollingerBounce": BollingerBounceStrategy,
    "Stochastic": StochasticStrategy,
    "WilliamsR": WilliamsRStrategy,
    "CCIReversal": CCIReversalStrategy,
    # Volatility
    "ATRBreakout": ATRBreakoutStrategy,
    "KeltnerBreakout": KeltnerBreakoutStrategy,
    "VolTargeting": VolTargetingStrategy,
    "BBSqueeze": BBSqueezeStrategy,
    # Volume
    "OBVConfirmation": OBVConfirmationStrategy,
    "MFIReversal": MFIReversalStrategy,
    "VWAPReversion": VWAPReversionStrategy,
    # Momentum
    "Momentum12-1": MomentumStrategy,
    "AroonTrend": AroonTrendStrategy,
    "TRIXTrend": TRIXTrendStrategy,
}

BASELINES = ["BuyAndHold", "SMA200", "GoldenCrossBaseline"]

CATEGORIES = {
    "baseline": BASELINES,
    "trend_following": [
        "GoldenCross", "MACDTrend", "ADXBreakout",
        "Ichimoku", "ParabolicSAR", "DonchianBreakout"
    ],
    "mean_reversion": [
        "RSIReversal", "BollingerBounce", "Stochastic",
        "WilliamsR", "CCIReversal"
    ],
    "volatility": [
        "ATRBreakout", "KeltnerBreakout", "VolTargeting", "BBSqueeze"
    ],
    "volume": ["OBVConfirmation", "MFIReversal", "VWAPReversion"],
    "momentum": ["Momentum12-1", "AroonTrend", "TRIXTrend"],
}


def load_data(start_date: str = "2000-01-01", end_date: str = None) -> pd.DataFrame:
    """Load and prepare QQQ data."""
    print("Loading QQQ data...")
    loader = QQQDataLoader(start_date=start_date, end_date=end_date)
    data = loader.fetch()

    # Add returns
    data = loader.add_returns()

    # Version the data
    versioner = DataVersioner()
    data_hash = versioner.compute_hash(data)
    print(f"Data hash: {data_hash[:16]}...")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    print(f"Total rows: {len(data)}")

    return data


def run_single_backtest(
    strategy_name: str,
    data: pd.DataFrame,
    config: BacktestConfig,
    regimes: pd.Series = None,
) -> dict:
    """Run backtest for a single strategy."""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_class = STRATEGIES[strategy_name]
    strategy = strategy_class()

    engine = BacktestEngine(config=config)
    result = engine.run(strategy, data, regimes=regimes)

    return {
        "strategy": strategy_name,
        "total_return": result.metrics.total_return,
        "annualized_return": result.metrics.annualized_return,
        "sharpe_ratio": result.metrics.sharpe_ratio,
        "sortino_ratio": result.metrics.sortino_ratio,
        "max_drawdown": result.metrics.max_drawdown,
        "calmar_ratio": result.metrics.calmar_ratio,
        "num_trades": result.metrics.num_trades,
        "win_rate": result.metrics.win_rate,
        "profit_factor": result.metrics.profit_factor,
        "total_costs": result.metrics.total_costs,
        "avg_holding_period": result.metrics.avg_holding_period,
    }


def run_backtests(
    strategy_names: list,
    data: pd.DataFrame,
    config: BacktestConfig,
    regimes: pd.Series = None,
) -> pd.DataFrame:
    """Run backtests for multiple strategies."""
    results = []

    for name in strategy_names:
        print(f"  Running {name}...")
        try:
            result = run_single_backtest(name, data, config, regimes)
            results.append(result)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"strategy": name, "error": str(e)})

    return pd.DataFrame(results)


def generate_report(results: pd.DataFrame, output_dir: Path) -> None:
    """Generate and save performance report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = output_dir / f"backtest_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)

    if "error" in results.columns:
        valid = results[results["error"].isna()]
        errors = results[results["error"].notna()]
    else:
        valid = results
        errors = pd.DataFrame()

    if len(valid) > 0:
        # Sort by Sharpe ratio
        valid_sorted = valid.sort_values("sharpe_ratio", ascending=False)

        print("\nTop 5 by Sharpe Ratio:")
        print("-" * 60)
        for _, row in valid_sorted.head().iterrows():
            print(f"  {row['strategy']:20} | Sharpe: {row['sharpe_ratio']:6.2f} | "
                  f"Return: {row['annualized_return']*100:6.1f}% | "
                  f"MaxDD: {row['max_drawdown']*100:6.1f}%")

        print("\nBaseline Comparison:")
        print("-" * 60)
        baseline_results = valid[valid["strategy"].isin(BASELINES)]
        for _, row in baseline_results.iterrows():
            print(f"  {row['strategy']:20} | Sharpe: {row['sharpe_ratio']:6.2f} | "
                  f"Return: {row['annualized_return']*100:6.1f}% | "
                  f"MaxDD: {row['max_drawdown']*100:6.1f}%")

        print("\nStrategies Beating Buy & Hold:")
        print("-" * 60)
        bh_sharpe = valid[valid["strategy"] == "BuyAndHold"]["sharpe_ratio"].values
        if len(bh_sharpe) > 0:
            bh_sharpe = bh_sharpe[0]
            beating = valid[valid["sharpe_ratio"] > bh_sharpe]
            print(f"  {len(beating)} strategies beat B&H (Sharpe > {bh_sharpe:.2f})")

    if len(errors) > 0:
        print(f"\nErrors: {len(errors)} strategies failed")
        for _, row in errors.iterrows():
            print(f"  {row['strategy']}: {row['error']}")


def main():
    parser = argparse.ArgumentParser(description="Run backtests on trading strategies")
    parser.add_argument("--all", action="store_true", help="Run all strategies")
    parser.add_argument("--baselines", action="store_true", help="Run baseline strategies only")
    parser.add_argument("--strategy", type=str, help="Run a specific strategy")
    parser.add_argument("--category", type=str, choices=list(CATEGORIES.keys()),
                        help="Run strategies in a category")
    parser.add_argument("--start-date", type=str, default="2000-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=500_000,
                        help="Initial capital")
    parser.add_argument("--max-leverage", type=float, default=3.0,
                        help="Maximum leverage")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--with-regimes", action="store_true",
                        help="Detect and include regime analysis")

    args = parser.parse_args()

    # Determine which strategies to run
    if args.all:
        strategies = list(STRATEGIES.keys())
    elif args.baselines:
        strategies = BASELINES
    elif args.strategy:
        strategies = [args.strategy]
    elif args.category:
        strategies = CATEGORIES[args.category]
    else:
        print("Please specify --all, --baselines, --strategy, or --category")
        parser.print_help()
        sys.exit(1)

    print(f"Running {len(strategies)} strategies...")

    # Load data
    data = load_data(args.start_date, args.end_date)

    # Detect regimes if requested
    regimes = None
    if args.with_regimes:
        print("Detecting regimes...")
        detector = RulesBasedDetector()
        regimes = detector.detect_all(data)
        print(f"Regime distribution:\n{regimes.value_counts()}")

    # Configure backtest
    config = BacktestConfig(
        initial_capital=args.capital,
        max_leverage=args.max_leverage,
        max_drawdown=0.25,  # 25% hard limit per charter
    )

    # Run backtests
    print("\nRunning backtests...")
    results = run_backtests(strategies, data, config, regimes)

    # Generate report
    output_dir = Path(args.output_dir)
    generate_report(results, output_dir)


if __name__ == "__main__":
    main()

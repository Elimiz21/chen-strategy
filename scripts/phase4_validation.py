#!/usr/bin/env python3
"""
Phase 4 Validation Script
=========================

Comprehensive validation for regime detection and strategy performance.

This script:
1. Runs walk-forward validation on high-Sharpe strategies
2. Calibrates regime detection on real QQQ data
3. Builds expert-regime performance matrix
4. Runs statistical tests (ANOVA) on strategy-regime relationships

Usage:
    python scripts/phase4_validation.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from scipy import stats

from data.loader import QQQDataLoader
from backtesting.engine import BacktestEngine, BacktestConfig, WalkForwardValidator
from backtesting.cost_model import CostModel
from regime.detector import (
    RulesBasedDetector,
    ThresholdDetector,
    HybridDetector,
    Regime,
    analyze_regime_persistence,
    compute_regime_transition_matrix,
)

# Import strategies
from strategies import (
    BuyAndHoldStrategy,
    SMA200Strategy,
    GoldenCrossBaselineStrategy,
    GoldenCrossStrategy,
    MACDTrendStrategy,
    ADXBreakoutStrategy,
    IchimokuStrategy,
    ParabolicSARStrategy,
    DonchianBreakoutStrategy,
    RSIReversalStrategy,
    BollingerBounceStrategy,
    StochasticStrategy,
    WilliamsRStrategy,
    CCIReversalStrategy,
    ATRBreakoutStrategy,
    KeltnerBreakoutStrategy,
    VolTargetingStrategy,
    BBSqueezeStrategy,
    OBVConfirmationStrategy,
    MFIReversalStrategy,
    VWAPReversionStrategy,
    MomentumStrategy,
    AroonTrendStrategy,
    TRIXTrendStrategy,
)


# High-Sharpe strategies to validate (from initial backtest)
HIGH_SHARPE_STRATEGIES = {
    "BBSqueeze": BBSqueezeStrategy,
    "DonchianBreakout": DonchianBreakoutStrategy,
    "ParabolicSAR": ParabolicSARStrategy,
    "KeltnerBreakout": KeltnerBreakoutStrategy,
    "Ichimoku": IchimokuStrategy,
    "ATRBreakout": ATRBreakoutStrategy,
    "OBVConfirmation": OBVConfirmationStrategy,
    "MACDTrend": MACDTrendStrategy,
}

# All strategies for regime analysis
ALL_STRATEGIES = {
    "BuyAndHold": BuyAndHoldStrategy,
    "SMA200": SMA200Strategy,
    "GoldenCrossBaseline": GoldenCrossBaselineStrategy,
    "GoldenCross": GoldenCrossStrategy,
    "MACDTrend": MACDTrendStrategy,
    "ADXBreakout": ADXBreakoutStrategy,
    "Ichimoku": IchimokuStrategy,
    "ParabolicSAR": ParabolicSARStrategy,
    "DonchianBreakout": DonchianBreakoutStrategy,
    "RSIReversal": RSIReversalStrategy,
    "BollingerBounce": BollingerBounceStrategy,
    "Stochastic": StochasticStrategy,
    "WilliamsR": WilliamsRStrategy,
    "CCIReversal": CCIReversalStrategy,
    "ATRBreakout": ATRBreakoutStrategy,
    "KeltnerBreakout": KeltnerBreakoutStrategy,
    "VolTargeting": VolTargetingStrategy,
    "BBSqueeze": BBSqueezeStrategy,
    "OBVConfirmation": OBVConfirmationStrategy,
    "MFIReversal": MFIReversalStrategy,
    "VWAPReversion": VWAPReversionStrategy,
    "Momentum12-1": MomentumStrategy,
    "AroonTrend": AroonTrendStrategy,
    "TRIXTrend": TRIXTrendStrategy,
}


def load_data(start_date: str = "2010-01-01", end_date: str = None) -> pd.DataFrame:
    """Load QQQ data with extended history for walk-forward."""
    print("Loading QQQ data...")
    loader = QQQDataLoader(start_date=start_date, end_date=end_date)
    data = loader.fetch()
    data = loader.add_returns()
    print(f"  Loaded {len(data)} trading days ({data.index[0].date()} to {data.index[-1].date()})")
    return data


def run_walk_forward_validation(
    data: pd.DataFrame,
    strategies: dict,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Run walk-forward validation on strategies.

    Uses simple train/test split: 2010-2019 train, 2020-2024 test.
    """
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (Train: 2010-2019, Test: 2020-2024)")
    print("=" * 80)

    # Simple time-based split
    split_date = "2020-01-01"
    train_data = data[data.index < split_date]
    test_data = data[data.index >= split_date]

    print(f"  Train: {len(train_data)} days ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    print(f"  Test: {len(test_data)} days ({test_data.index[0].date()} to {test_data.index[-1].date()})")

    engine = BacktestEngine(config=config)
    results = []

    for name, strategy_class in strategies.items():
        print(f"\n  Validating {name}...")
        strategy = strategy_class()

        try:
            # In-sample (train)
            train_result = engine.run(strategy, train_data)
            is_sharpe = train_result.metrics.sharpe_ratio

            # Out-of-sample (test)
            test_result = engine.run(strategy, test_data)
            oos_sharpe = test_result.metrics.sharpe_ratio

            # Calculate metrics
            sharpe_decay = is_sharpe - oos_sharpe
            overfit_ratio = is_sharpe / oos_sharpe if oos_sharpe != 0 else float("inf")

            results.append({
                "strategy": name,
                "avg_is_sharpe": is_sharpe,
                "avg_oos_sharpe": oos_sharpe,
                "avg_sharpe_decay": sharpe_decay,
                "oos_sharpe_std": 0,  # Single fold
                "overfit_ratio": overfit_ratio,
                "n_folds": 1,
                "is_return": train_result.metrics.annualized_return,
                "oos_return": test_result.metrics.annualized_return,
                "is_max_dd": train_result.metrics.max_drawdown,
                "oos_max_dd": test_result.metrics.max_drawdown,
            })

            print(f"    IS Sharpe: {is_sharpe:.2f} | OOS Sharpe: {oos_sharpe:.2f}")
            print(f"    IS Return: {train_result.metrics.annualized_return*100:.1f}% | OOS Return: {test_result.metrics.annualized_return*100:.1f}%")
            print(f"    Sharpe Decay: {sharpe_decay:.2f} | Overfit Ratio: {overfit_ratio:.2f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "strategy": name,
                "error": str(e),
            })

    return pd.DataFrame(results)


def calibrate_regime_detection(data: pd.DataFrame) -> dict:
    """
    Calibrate and evaluate regime detection methods.

    Returns accuracy metrics and regime statistics.
    """
    print("\n" + "=" * 80)
    print("REGIME DETECTION CALIBRATION")
    print("=" * 80)

    detectors = {
        "RulesBased": RulesBasedDetector(),
        "Threshold": ThresholdDetector(),
        "Hybrid": HybridDetector(),
    }

    results = {}

    for name, detector in detectors.items():
        print(f"\n  Calibrating {name}...")

        # Detect regimes for all data
        regimes = detector.detect_all(data, start_idx=252)

        # Analyze persistence
        persistence = analyze_regime_persistence(regimes)

        # Compute transition matrix
        trans_matrix, states = compute_regime_transition_matrix(regimes)

        # Regime distribution
        regime_counts = regimes.value_counts()
        regime_pct = (regime_counts / len(regimes) * 100).to_dict()

        # Calculate stability (how often regime stays the same)
        stability = (regimes == regimes.shift(1)).mean()

        results[name] = {
            "regime_distribution": regime_pct,
            "persistence": persistence,
            "stability": stability,
            "n_transitions": (regimes != regimes.shift(1)).sum(),
            "unique_regimes": len(regime_counts),
        }

        print(f"    Stability: {stability:.1%}")
        print(f"    Transitions: {results[name]['n_transitions']}")
        print(f"    Regime Distribution:")
        for regime, pct in sorted(regime_pct.items(), key=lambda x: -x[1])[:5]:
            print(f"      {regime}: {pct:.1f}%")

    return results


def build_expert_regime_matrix(
    data: pd.DataFrame,
    strategies: dict,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Build performance matrix: strategy returns by regime.

    Returns DataFrame with strategies as rows, regimes as columns.
    """
    print("\n" + "=" * 80)
    print("EXPERT-REGIME PERFORMANCE MATRIX")
    print("=" * 80)

    # Use rules-based detector as primary
    detector = RulesBasedDetector()
    regimes = detector.detect_all(data, start_idx=252)

    engine = BacktestEngine(config=config)

    # Collect returns by strategy and regime
    matrix_data = []
    strategy_returns_by_regime = {}

    for name, strategy_class in strategies.items():
        print(f"  Analyzing {name}...")
        strategy = strategy_class()

        try:
            result = engine.run(strategy, data, regimes=regimes)

            # Calculate returns by regime
            returns = result.returns
            regime_returns = {}
            regime_returns_list = {}

            for regime in regimes.unique():
                if pd.isna(regime):
                    continue

                mask = regimes == regime
                regime_ret = returns[mask]

                if len(regime_ret) > 20:  # Minimum sample
                    ann_ret = (1 + regime_ret.mean()) ** 252 - 1
                    regime_returns[regime] = ann_ret
                    regime_returns_list[regime] = regime_ret.values
                else:
                    regime_returns[regime] = np.nan
                    regime_returns_list[regime] = []

            strategy_returns_by_regime[name] = regime_returns_list

            row = {"strategy": name}
            row.update(regime_returns)
            row["overall_sharpe"] = result.metrics.sharpe_ratio
            matrix_data.append(row)

        except Exception as e:
            print(f"    ERROR: {e}")

    matrix_df = pd.DataFrame(matrix_data).set_index("strategy")

    # Print top performers by regime
    print("\n  Top performers by regime:")
    regime_cols = [c for c in matrix_df.columns if c != "overall_sharpe"]

    for regime in regime_cols:
        col = matrix_df[regime].dropna()
        if len(col) > 0:
            best = col.idxmax()
            print(f"    {regime}: {best} ({col[best]*100:.1f}% ann.)")

    return matrix_df, strategy_returns_by_regime


def run_anova_tests(strategy_returns_by_regime: dict) -> pd.DataFrame:
    """
    Run ANOVA tests to determine if strategy performance differs by regime.

    Returns DataFrame with F-statistic and p-value for each strategy.
    """
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS (ANOVA)")
    print("=" * 80)
    print("\nTesting: Do strategy returns significantly differ across regimes?")
    print("Null hypothesis: Returns are the same across all regimes")
    print("Alternative: Returns differ by regime (regime awareness helps)")

    results = []

    for strategy, regime_returns in strategy_returns_by_regime.items():
        # Get non-empty groups
        groups = [np.array(returns) for returns in regime_returns.values()
                  if len(returns) > 20]

        if len(groups) < 2:
            results.append({
                "strategy": strategy,
                "f_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "n_regimes": len(groups),
            })
            continue

        try:
            f_stat, p_value = stats.f_oneway(*groups)

            results.append({
                "strategy": strategy,
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "n_regimes": len(groups),
            })
        except Exception as e:
            results.append({
                "strategy": strategy,
                "f_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
                "error": str(e),
            })

    results_df = pd.DataFrame(results)

    # Summary
    significant_count = results_df["significant"].sum()
    total = len(results_df[results_df["p_value"].notna()])

    print(f"\n  Results: {significant_count}/{total} strategies show regime-dependent performance (p < 0.05)")

    # Show most regime-dependent strategies
    print("\n  Most regime-dependent strategies (lowest p-values):")
    sorted_results = results_df.dropna(subset=["p_value"]).sort_values("p_value")
    for _, row in sorted_results.head(10).iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
        print(f"    {row['strategy']:20} F={row['f_statistic']:8.2f}  p={row['p_value']:.4f} {sig}")

    return results_df


def generate_phase4_report(
    wf_results: pd.DataFrame,
    regime_calibration: dict,
    regime_matrix: pd.DataFrame,
    anova_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate Phase 4 validation report."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV results
    wf_results.to_csv(output_dir / f"walk_forward_validation_{timestamp}.csv", index=False)
    regime_matrix.to_csv(output_dir / f"regime_matrix_{timestamp}.csv")
    anova_results.to_csv(output_dir / f"anova_results_{timestamp}.csv", index=False)

    # Save JSON calibration
    with open(output_dir / f"regime_calibration_{timestamp}.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(regime_calibration, f, indent=2, default=convert)

    # Generate summary report
    report = []
    report.append("=" * 80)
    report.append("PHASE 4 VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 80)

    # Walk-forward summary
    report.append("\n## WALK-FORWARD VALIDATION SUMMARY\n")

    valid_wf = wf_results[wf_results["avg_oos_sharpe"].notna()]
    if len(valid_wf) > 0:
        # Strategies that maintain performance OOS
        robust = valid_wf[valid_wf["overfit_ratio"] < 2.0]
        report.append(f"Strategies with overfit ratio < 2.0: {len(robust)}/{len(valid_wf)}")

        # Sort by OOS Sharpe
        sorted_wf = valid_wf.sort_values("avg_oos_sharpe", ascending=False)
        report.append("\nTop 5 by Out-of-Sample Sharpe:")
        for _, row in sorted_wf.head().iterrows():
            report.append(f"  {row['strategy']:20} OOS Sharpe: {row['avg_oos_sharpe']:.2f} (Decay: {row['avg_sharpe_decay']:.2f})")

    # Regime detection summary
    report.append("\n## REGIME DETECTION SUMMARY\n")
    for detector, metrics in regime_calibration.items():
        report.append(f"\n{detector}:")
        report.append(f"  Stability: {metrics['stability']:.1%}")
        report.append(f"  Transitions: {metrics['n_transitions']}")

    # ANOVA summary
    report.append("\n## REGIME-DEPENDENCE ANALYSIS (ANOVA)\n")
    significant = anova_results[anova_results["significant"] == True]
    report.append(f"Strategies with regime-dependent performance: {len(significant)}/{len(anova_results)}")

    if len(significant) > 0:
        report.append("\nRegime-dependent strategies (p < 0.05):")
        for _, row in significant.sort_values("p_value").iterrows():
            report.append(f"  {row['strategy']:20} (p = {row['p_value']:.4f})")

    # Conclusions
    report.append("\n## PHASE 4 CONCLUSIONS\n")

    # Check phase pass criteria
    regime_accuracy_ok = any(m["stability"] > 0.8 for m in regime_calibration.values())
    regime_dependent_ok = len(significant) >= len(anova_results) * 0.3
    oos_viable = len(valid_wf[valid_wf["avg_oos_sharpe"] > 0.5]) >= 3

    report.append(f"1. Regime detection stability > 80%: {'PASS' if regime_accuracy_ok else 'FAIL'}")
    report.append(f"2. >30% strategies regime-dependent: {'PASS' if regime_dependent_ok else 'FAIL'}")
    report.append(f"3. >=3 strategies with OOS Sharpe > 0.5: {'PASS' if oos_viable else 'FAIL'}")

    all_pass = regime_accuracy_ok and regime_dependent_ok and oos_viable
    report.append(f"\nPHASE 4 OVERALL: {'PASS' if all_pass else 'NEEDS REVIEW'}")

    # Write report
    report_text = "\n".join(report)
    with open(output_dir / f"phase4_report_{timestamp}.txt", "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\n\nResults saved to: {output_dir}")


def main():
    print("=" * 80)
    print("CHEN STRATEGY - PHASE 4 VALIDATION")
    print("Regime Detection Calibration & Walk-Forward Validation")
    print("=" * 80)

    # Load data (extended history for walk-forward)
    data = load_data(start_date="2010-01-01")

    # Configuration
    config = BacktestConfig(
        initial_capital=500_000,
        max_leverage=3.0,
        max_drawdown=0.25,
        warmup_period=252,  # 1 year warmup for walk-forward
    )

    # Output directory
    output_dir = Path("results/phase4")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Walk-forward validation on high-Sharpe strategies
    wf_results = run_walk_forward_validation(data, HIGH_SHARPE_STRATEGIES, config)

    # 2. Calibrate regime detection
    regime_calibration = calibrate_regime_detection(data)

    # 3. Build expert-regime matrix
    regime_matrix, strategy_returns_by_regime = build_expert_regime_matrix(
        data, ALL_STRATEGIES, config
    )

    # 4. ANOVA tests
    anova_results = run_anova_tests(strategy_returns_by_regime)

    # 5. Generate report
    generate_phase4_report(
        wf_results,
        regime_calibration,
        regime_matrix,
        anova_results,
        output_dir,
    )

    print("\n" + "=" * 80)
    print("PHASE 4 VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

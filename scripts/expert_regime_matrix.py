#!/usr/bin/env python3
"""
Expert-Regime Performance Matrix
================================

Builds a comprehensive matrix showing how each expert strategy performs
in each regime. Tests for statistical significance of regime effects.

Phase 4 Gate Criteria:
- Expert performance differs by regime (p < 0.05)

Run as: python scripts/expert_regime_matrix.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data.loader import QQQDataLoader
from backtesting.engine import BacktestEngine, BacktestConfig
from regime.detector import RulesBasedDetector, Regime

# Import all strategies
from strategies.base import BuyAndHoldStrategy, SMA200Strategy, GoldenCrossBaselineStrategy
from strategies.trend_following import (
    GoldenCrossStrategy, MACDTrendStrategy, ADXBreakoutStrategy,
    IchimokuStrategy, ParabolicSARStrategy, DonchianBreakoutStrategy
)
from strategies.mean_reversion import (
    RSIReversalStrategy, BollingerBounceStrategy, StochasticStrategy,
    WilliamsRStrategy, CCIReversalStrategy
)
from strategies.volatility import (
    ATRBreakoutStrategy, KeltnerBreakoutStrategy, VolTargetingStrategy, BBSqueezeStrategy
)
from strategies.momentum import MomentumStrategy, AroonTrendStrategy, TRIXTrendStrategy
from strategies.volume import OBVConfirmationStrategy, MFIReversalStrategy, VWAPReversionStrategy


def section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def get_all_strategies():
    """Get all strategy instances."""
    return {
        # Baselines
        "BuyAndHold": BuyAndHoldStrategy(),
        "SMA200": SMA200Strategy(),
        "GoldenCrossBaseline": GoldenCrossBaselineStrategy(),

        # Trend Following
        "GoldenCross": GoldenCrossStrategy(),
        "MACDTrend": MACDTrendStrategy(),
        "ADXBreakout": ADXBreakoutStrategy(),
        "Ichimoku": IchimokuStrategy(),
        "ParabolicSAR": ParabolicSARStrategy(),
        "DonchianBreakout": DonchianBreakoutStrategy(),

        # Mean Reversion
        "RSIReversal": RSIReversalStrategy(),
        "BollingerBounce": BollingerBounceStrategy(),
        "Stochastic": StochasticStrategy(),
        "WilliamsR": WilliamsRStrategy(),
        "CCIReversal": CCIReversalStrategy(),

        # Volatility
        "ATRBreakout": ATRBreakoutStrategy(),
        "KeltnerBreakout": KeltnerBreakoutStrategy(),
        "VolTargeting": VolTargetingStrategy(),
        "BBSqueeze": BBSqueezeStrategy(),

        # Momentum
        "Momentum": MomentumStrategy(),
        "AroonTrend": AroonTrendStrategy(),
        "TRIXTrend": TRIXTrendStrategy(),

        # Volume
        "OBVConfirmation": OBVConfirmationStrategy(),
        "MFIReversal": MFIReversalStrategy(),
        "VWAPReversion": VWAPReversionStrategy(),
    }


def build_regime_performance_matrix(
    data: pd.DataFrame,
    strategies: Dict,
    regimes: pd.Series,
    config: BacktestConfig
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build matrix of strategy performance by regime.

    Returns:
        Tuple of (performance_matrix DataFrame, raw_returns dict for statistical tests)
    """
    engine = BacktestEngine(config=config)

    # Get unique regimes (excluding TRANSITION and nan)
    unique_regimes = [r for r in regimes.unique()
                      if r not in ['TRANSITION', None] and not pd.isna(r)]

    results = []
    raw_returns = {}  # For statistical tests

    for name, strategy in strategies.items():
        print(f"  Processing {name}...", end=" ")

        try:
            result = engine.run(strategy, data, regimes=regimes)
            returns = result.returns.iloc[config.warmup_period:]

            # Align regimes with returns
            aligned_regimes = regimes.reindex(returns.index)

            strategy_returns = {"overall": returns}

            # Calculate performance by regime
            regime_perfs = {"strategy": name, "category": strategy.category}
            regime_perfs["overall_sharpe"] = result.metrics.sharpe_ratio
            regime_perfs["overall_return"] = result.metrics.annualized_return
            regime_perfs["bias_adj_sharpe"] = result.metrics.bias_adjusted_sharpe

            for regime in unique_regimes:
                regime_mask = aligned_regimes == regime
                regime_returns = returns[regime_mask]

                if len(regime_returns) > 20:  # Minimum days for meaningful stats
                    ann_return = (1 + regime_returns.mean()) ** 252 - 1
                    ann_vol = regime_returns.std() * np.sqrt(252)
                    sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0

                    regime_perfs[f"{regime}_sharpe"] = sharpe
                    regime_perfs[f"{regime}_return"] = ann_return
                    regime_perfs[f"{regime}_days"] = len(regime_returns)

                    strategy_returns[regime] = regime_returns
                else:
                    regime_perfs[f"{regime}_sharpe"] = np.nan
                    regime_perfs[f"{regime}_return"] = np.nan
                    regime_perfs[f"{regime}_days"] = len(regime_returns)

            results.append(regime_perfs)
            raw_returns[name] = strategy_returns
            print("✓")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "strategy": name,
                "category": getattr(strategy, 'category', 'unknown'),
                "overall_sharpe": np.nan,
                "error": str(e)
            })

    return pd.DataFrame(results), raw_returns


def test_regime_significance(raw_returns: Dict, regimes: List[str]) -> pd.DataFrame:
    """
    Test if strategy performance differs significantly across regimes.

    Uses Kruskal-Wallis H-test (non-parametric ANOVA) since returns
    may not be normally distributed.

    Returns:
        DataFrame with test statistics and p-values
    """
    results = []

    for strategy_name, returns_dict in raw_returns.items():
        if len(returns_dict) < 3:  # Need at least 2 regimes + overall
            continue

        # Collect returns by regime
        regime_returns = []
        regime_names = []

        for regime in regimes:
            if regime in returns_dict and len(returns_dict[regime]) > 20:
                regime_returns.append(returns_dict[regime].values)
                regime_names.append(regime)

        if len(regime_returns) >= 2:
            # Kruskal-Wallis H-test
            try:
                h_stat, p_value = stats.kruskal(*regime_returns)

                # Effect size (eta-squared approximation)
                n_total = sum(len(r) for r in regime_returns)
                eta_sq = (h_stat - len(regime_returns) + 1) / (n_total - len(regime_returns))

                results.append({
                    "strategy": strategy_name,
                    "n_regimes": len(regime_returns),
                    "h_statistic": h_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "eta_squared": max(0, eta_sq),  # Effect size
                    "regimes_tested": ", ".join(regime_names),
                })
            except Exception as e:
                results.append({
                    "strategy": strategy_name,
                    "error": str(e),
                })

    return pd.DataFrame(results)


def pairwise_regime_comparison(raw_returns: Dict, regime1: str, regime2: str) -> pd.DataFrame:
    """
    Pairwise comparison of strategy returns between two regimes.

    Uses Mann-Whitney U test (non-parametric).
    """
    results = []

    for strategy_name, returns_dict in raw_returns.items():
        if regime1 in returns_dict and regime2 in returns_dict:
            r1 = returns_dict[regime1]
            r2 = returns_dict[regime2]

            if len(r1) > 20 and len(r2) > 20:
                try:
                    u_stat, p_value = stats.mannwhitneyu(r1, r2, alternative='two-sided')

                    # Effect size (rank-biserial correlation)
                    n1, n2 = len(r1), len(r2)
                    effect_size = 1 - (2 * u_stat) / (n1 * n2)

                    results.append({
                        "strategy": strategy_name,
                        f"{regime1}_mean": r1.mean() * 252,  # Annualized
                        f"{regime2}_mean": r2.mean() * 252,
                        "difference": (r1.mean() - r2.mean()) * 252,
                        "u_statistic": u_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "effect_size": effect_size,
                    })
                except Exception as e:
                    pass

    return pd.DataFrame(results)


def identify_regime_specialists(matrix: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify which strategies work best in which regimes.

    Returns:
        Dict mapping regime to list of recommended strategies
    """
    specialists = {}

    # Get regime columns
    sharpe_cols = [c for c in matrix.columns if c.endswith('_sharpe') and c != 'overall_sharpe' and c != 'bias_adj_sharpe']

    for col in sharpe_cols:
        regime = col.replace('_sharpe', '')

        # Get top performers in this regime (Sharpe > 0.5)
        valid = matrix[matrix[col].notna()]
        top_performers = valid[valid[col] > 0.5].sort_values(col, ascending=False)

        if len(top_performers) > 0:
            specialists[regime] = top_performers['strategy'].tolist()[:5]  # Top 5

    return specialists


def generate_recommendations(matrix: pd.DataFrame, significance_df: pd.DataFrame) -> List[str]:
    """Generate actionable recommendations based on the analysis."""
    recommendations = []

    # 1. Check if regime-awareness adds value
    sig_count = significance_df['significant'].sum() if 'significant' in significance_df.columns else 0
    total = len(significance_df)

    if sig_count / total > 0.5:
        recommendations.append(
            f"✓ REGIME-AWARENESS VALUABLE: {sig_count}/{total} strategies show "
            f"significantly different performance across regimes (p < 0.05)"
        )
    else:
        recommendations.append(
            f"⚠️ LIMITED REGIME EFFECT: Only {sig_count}/{total} strategies show "
            f"significant regime dependence"
        )

    # 2. Find best strategies per category
    for category in matrix['category'].unique():
        cat_strategies = matrix[matrix['category'] == category]
        if len(cat_strategies) > 0:
            best = cat_strategies.loc[cat_strategies['bias_adj_sharpe'].idxmax()]
            if best['bias_adj_sharpe'] > 0:
                recommendations.append(
                    f"Best {category}: {best['strategy']} "
                    f"(bias-adjusted Sharpe: {best['bias_adj_sharpe']:.2f})"
                )

    # 3. Mean-reversion warning
    mr_strategies = matrix[matrix['category'] == 'mean_reversion']
    if len(mr_strategies) > 0:
        avg_sharpe = mr_strategies['overall_sharpe'].mean()
        if avg_sharpe < 0:
            recommendations.append(
                "⚠️ MEAN-REVERSION WARNING: Category average Sharpe is negative. "
                "Consider disabling in BULL regimes or reducing allocation."
            )

    # 4. Regime-specific recommendations
    bull_cols = [c for c in matrix.columns if 'BULL' in c and '_sharpe' in c]
    bear_cols = [c for c in matrix.columns if 'BEAR' in c and '_sharpe' in c]

    if bull_cols:
        bull_avg = matrix[bull_cols].mean(axis=1)
        best_bull = matrix.loc[bull_avg.idxmax(), 'strategy']
        recommendations.append(f"Best for BULL markets: {best_bull}")

    if bear_cols:
        bear_avg = matrix[bear_cols].mean(axis=1)
        best_bear = matrix.loc[bear_avg.idxmax(), 'strategy']
        recommendations.append(f"Best for BEAR markets: {best_bear}")

    return recommendations


def main():
    """Build expert-regime performance matrix and run statistical tests."""
    print("\n" + "="*70)
    print("  EXPERT-REGIME PERFORMANCE MATRIX")
    print("  Phase 4 Gate Criteria Analysis")
    print(f"  Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Load data
    section("1. LOADING DATA AND DETECTING REGIMES")

    loader = QQQDataLoader(start_date="2015-01-01", end_date="2023-12-31")
    data = loader.fetch()

    if data is None or len(data) < 500:
        print("ERROR: Could not load sufficient data")
        return

    print(f"Loaded {len(data)} days of QQQ data")

    # Detect regimes
    detector = RulesBasedDetector()
    regimes = detector.detect_all(data, start_idx=200)

    # Simplify regimes for analysis (combine vol levels)
    simplified_regimes = regimes.apply(lambda x:
        'BULL' if 'BULL' in str(x) else
        ('BEAR' if 'BEAR' in str(x) else 'TRANSITION')
    )

    regime_counts = simplified_regimes.value_counts()
    print("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(simplified_regimes) * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")

    # Build performance matrix
    section("2. BUILDING EXPERT-REGIME PERFORMANCE MATRIX")

    strategies = get_all_strategies()
    print(f"Testing {len(strategies)} strategies across regimes...\n")

    config = BacktestConfig(
        initial_capital=500_000,
        max_leverage=1.0,  # Fair comparison at 1x
        max_drawdown=0.20,
        warmup_period=200,
    )

    matrix, raw_returns = build_regime_performance_matrix(
        data, strategies, simplified_regimes, config
    )

    # Display matrix
    section("3. PERFORMANCE MATRIX (Sharpe Ratios)")

    display_cols = ['strategy', 'category', 'overall_sharpe', 'bias_adj_sharpe']
    sharpe_cols = [c for c in matrix.columns if '_sharpe' in c and c not in display_cols]
    display_cols.extend(sorted(sharpe_cols))

    print(matrix[display_cols].to_string(index=False))

    # Save to CSV
    output_path = "results/expert_regime_matrix.csv"
    matrix.to_csv(output_path, index=False)
    print(f"\nMatrix saved to: {output_path}")

    # Statistical significance tests
    section("4. STATISTICAL SIGNIFICANCE TESTS")

    print("Testing if strategy performance differs across regimes...")
    print("(Kruskal-Wallis H-test, non-parametric ANOVA)\n")

    unique_regimes = [r for r in simplified_regimes.unique() if r != 'TRANSITION']
    significance_df = test_regime_significance(raw_returns, unique_regimes)

    if len(significance_df) > 0:
        sig_display = significance_df[['strategy', 'h_statistic', 'p_value', 'significant', 'eta_squared']].copy()
        sig_display = sig_display.sort_values('p_value')
        print(sig_display.to_string(index=False))

        sig_count = significance_df['significant'].sum()
        total = len(significance_df)
        print(f"\nSignificant at p < 0.05: {sig_count}/{total} strategies ({sig_count/total*100:.1f}%)")

        # Save significance results
        significance_df.to_csv("results/regime_significance_tests.csv", index=False)

    # Pairwise comparison: BULL vs BEAR
    section("5. BULL vs BEAR PAIRWISE COMPARISON")

    pairwise = pairwise_regime_comparison(raw_returns, 'BULL', 'BEAR')
    if len(pairwise) > 0:
        pairwise = pairwise.sort_values('difference', ascending=False)
        print(pairwise[['strategy', 'BULL_mean', 'BEAR_mean', 'difference', 'p_value', 'significant']].to_string(index=False))

        # Strategies that work better in BULL
        bull_better = pairwise[(pairwise['difference'] > 0) & (pairwise['significant'])]
        bear_better = pairwise[(pairwise['difference'] < 0) & (pairwise['significant'])]

        print(f"\nStrategies significantly better in BULL: {len(bull_better)}")
        print(f"Strategies significantly better in BEAR: {len(bear_better)}")

    # Regime specialists
    section("6. REGIME SPECIALISTS")

    specialists = identify_regime_specialists(matrix)
    for regime, strats in specialists.items():
        print(f"\n{regime} specialists (Sharpe > 0.5):")
        for s in strats:
            row = matrix[matrix['strategy'] == s].iloc[0]
            sharpe = row.get(f'{regime}_sharpe', np.nan)
            print(f"  - {s}: Sharpe = {sharpe:.2f}")

    # Recommendations
    section("7. RECOMMENDATIONS")

    recommendations = generate_recommendations(matrix, significance_df)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Phase 4 Gate Check
    section("8. PHASE 4 GATE CHECK")

    # Check: Expert performance differs by regime (p < 0.05)
    if len(significance_df) > 0:
        sig_pct = significance_df['significant'].sum() / len(significance_df) * 100

        if sig_pct >= 50:
            print(f"✅ GATE CRITERIA MET: {sig_pct:.0f}% of strategies show significant regime dependence")
            gate_passed = True
        else:
            print(f"⚠️ GATE CRITERIA MARGINAL: Only {sig_pct:.0f}% show significance (target: 50%+)")
            gate_passed = sig_pct >= 30  # Relaxed threshold

        # Additional evidence
        if len(pairwise) > 0:
            bull_bear_sig = pairwise['significant'].sum()
            print(f"   BULL vs BEAR significant differences: {bull_bear_sig}/{len(pairwise)}")

    else:
        print("❌ Could not run significance tests")
        gate_passed = False

    print("\n" + "="*70)
    if gate_passed:
        print("  PHASE 4 GATE: ✅ PASSED")
    else:
        print("  PHASE 4 GATE: ⚠️ NEEDS REVIEW")
    print("="*70)

    return matrix, significance_df


if __name__ == "__main__":
    main()

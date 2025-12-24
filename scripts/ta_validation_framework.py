#!/usr/bin/env python3
"""
TA Validation Framework
=======================

Rigorous statistical validation of technical analysis strategies.

Based on Expert Panel recommendations:
1. Monte Carlo permutation testing
2. Multiple hypothesis correction (Bonferroni, FDR)
3. Strategy correlation analysis
4. Effective strategy count (PCA)
5. Regime-conditional testing

References:
- White (2000): "A Reality Check for Data Snooping"
- Hansen (2005): "A Test for Superior Predictive Ability"
- López de Prado (2018): "Advances in Financial Machine Learning"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data.loader import QQQDataLoader
from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.cost_model import CostModel


@dataclass
class ValidationResult:
    """Result of strategy validation."""
    strategy_name: str
    raw_sharpe: float
    raw_return: float
    p_value_mc: float  # Monte Carlo p-value
    p_value_adjusted: float  # Bonferroni adjusted
    significant: bool  # After multiple testing correction
    confidence_interval: Tuple[float, float]  # 95% CI for Sharpe
    effective_n: float  # Effective number of independent observations
    regime_stability: float  # Consistency across regimes
    correlation_with_market: float  # Beta


class MonteCarloValidator:
    """
    Monte Carlo permutation testing for strategy validation.

    Tests null hypothesis: Strategy return = random timing with same trade frequency

    Method:
    1. Generate N random price series with same statistical properties
    2. Run strategy on each random series
    3. Calculate p-value as % of random runs that beat real performance
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        block_size: int = 20,  # For block bootstrap to preserve autocorrelation
    ):
        self.n_simulations = n_simulations
        self.block_size = block_size

    def generate_random_prices(
        self, original: pd.Series, method: str = "block_bootstrap"
    ) -> pd.Series:
        """Generate random price series with similar properties."""
        returns = original.pct_change().dropna()

        if method == "block_bootstrap":
            # Block bootstrap to preserve autocorrelation
            n_blocks = len(returns) // self.block_size + 1
            random_returns = []

            for _ in range(n_blocks):
                start = np.random.randint(0, len(returns) - self.block_size)
                block = returns.iloc[start:start + self.block_size].values
                random_returns.extend(block)

            random_returns = random_returns[:len(returns)]

        elif method == "shuffle":
            # Simple shuffle (destroys autocorrelation)
            random_returns = np.random.permutation(returns.values)

        elif method == "gaussian":
            # Gaussian with matched mean/std
            random_returns = np.random.normal(
                returns.mean(), returns.std(), len(returns)
            )

        # Reconstruct prices
        random_prices = [original.iloc[0]]
        for ret in random_returns:
            random_prices.append(random_prices[-1] * (1 + ret))

        return pd.Series(random_prices, index=original.index[:len(random_prices)])

    def validate(
        self,
        strategy,
        data: pd.DataFrame,
        engine: BacktestEngine,
        metric: str = "sharpe_ratio",
    ) -> Dict:
        """
        Run Monte Carlo validation for a strategy.

        Returns:
            Dictionary with p-value, confidence interval, and distribution
        """
        # Run real backtest
        real_result = engine.run(strategy, data)
        real_metric = getattr(real_result.metrics, metric)

        # Run Monte Carlo simulations
        random_metrics = []
        for i in range(self.n_simulations):
            # Generate random price series
            random_close = self.generate_random_prices(data["close"])

            # Create random data
            random_data = data.copy()
            random_data["close"] = random_close

            # Regenerate OHLCV relationships
            price_ratio = random_close / data["close"]
            random_data["open"] = data["open"] * price_ratio
            random_data["high"] = data["high"] * price_ratio
            random_data["low"] = data["low"] * price_ratio

            try:
                # Reset strategy state if needed
                if hasattr(strategy, '_last_rebalance_idx'):
                    strategy._last_rebalance_idx = -999

                random_result = engine.run(strategy, random_data)
                random_metric = getattr(random_result.metrics, metric)
                random_metrics.append(random_metric)
            except Exception:
                continue  # Skip failed simulations

        random_metrics = np.array(random_metrics)

        # Calculate p-value (one-sided: how often does random beat real?)
        p_value = np.mean(random_metrics >= real_metric)

        # Calculate confidence interval for real metric
        # Using bootstrap of real returns
        real_returns = real_result.returns.dropna()
        bootstrap_metrics = []
        for _ in range(1000):
            bootstrap_returns = np.random.choice(
                real_returns, size=len(real_returns), replace=True
            )
            if metric == "sharpe_ratio":
                bootstrap_metric = (
                    np.mean(bootstrap_returns) * 252 - 0.04
                ) / (np.std(bootstrap_returns) * np.sqrt(252))
            else:
                bootstrap_metric = np.sum(bootstrap_returns)
            bootstrap_metrics.append(bootstrap_metric)

        ci_lower = np.percentile(bootstrap_metrics, 2.5)
        ci_upper = np.percentile(bootstrap_metrics, 97.5)

        return {
            "real_metric": real_metric,
            "p_value": p_value,
            "random_mean": np.mean(random_metrics),
            "random_std": np.std(random_metrics),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "percentile": np.mean(random_metrics < real_metric) * 100,
            "n_simulations": len(random_metrics),
        }


class MultipleTestingCorrector:
    """
    Correct for multiple hypothesis testing.

    Methods:
    - Bonferroni: p_adj = p * n_tests (very conservative)
    - Holm-Bonferroni: step-down procedure
    - Benjamini-Hochberg: FDR control
    """

    @staticmethod
    def bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
        """Apply Bonferroni correction."""
        n = len(p_values)
        return {k: min(1.0, v * n) for k, v in p_values.items()}

    @staticmethod
    def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
        """Apply Holm-Bonferroni step-down correction."""
        n = len(p_values)
        sorted_items = sorted(p_values.items(), key=lambda x: x[1])

        adjusted = {}
        for i, (name, p) in enumerate(sorted_items):
            adjusted[name] = min(1.0, p * (n - i))

        return adjusted

    @staticmethod
    def benjamini_hochberg(
        p_values: Dict[str, float], fdr: float = 0.05
    ) -> Dict[str, Tuple[float, bool]]:
        """
        Apply Benjamini-Hochberg FDR correction.

        Returns adjusted p-values and significance at given FDR level.
        """
        n = len(p_values)
        sorted_items = sorted(p_values.items(), key=lambda x: x[1])

        # Find largest i where p(i) <= i/n * FDR
        threshold_idx = 0
        for i, (name, p) in enumerate(sorted_items, 1):
            if p <= (i / n) * fdr:
                threshold_idx = i

        adjusted = {}
        for i, (name, p) in enumerate(sorted_items, 1):
            adj_p = p * n / i
            is_significant = i <= threshold_idx
            adjusted[name] = (min(1.0, adj_p), is_significant)

        return adjusted


class StrategyCorrelationAnalyzer:
    """
    Analyze correlations between strategy returns.

    Identifies:
    1. Highly correlated strategies (redundant)
    2. Negatively correlated strategies (diversifiers)
    3. Effective number of independent strategies (via PCA)
    """

    def __init__(self, min_correlation: float = 0.7):
        self.min_correlation = min_correlation

    def analyze(
        self, returns_dict: Dict[str, pd.Series]
    ) -> Dict:
        """
        Analyze strategy return correlations.

        Args:
            returns_dict: Dictionary mapping strategy name to return series

        Returns:
            Correlation analysis results
        """
        # Build returns matrix
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if len(returns_df) < 50:
            return {"error": "Insufficient data for correlation analysis"}

        # Correlation matrix
        corr_matrix = returns_df.corr()

        # Find highly correlated pairs
        redundant_pairs = []
        for i, s1 in enumerate(returns_df.columns):
            for j, s2 in enumerate(returns_df.columns):
                if i < j:
                    corr = corr_matrix.loc[s1, s2]
                    if abs(corr) > self.min_correlation:
                        redundant_pairs.append((s1, s2, corr))

        # Find diversifiers (negative correlation)
        diversifiers = []
        for i, s1 in enumerate(returns_df.columns):
            for j, s2 in enumerate(returns_df.columns):
                if i < j:
                    corr = corr_matrix.loc[s1, s2]
                    if corr < -0.3:
                        diversifiers.append((s1, s2, corr))

        # Effective number of strategies (PCA)
        eigenvalues = np.linalg.eigvals(corr_matrix.values)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 0]

        # Effective N = (sum of eigenvalues)^2 / sum of eigenvalues^2
        effective_n = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

        # Variance explained by top components
        sorted_eigs = np.sort(eigenvalues)[::-1]
        cum_var_explained = np.cumsum(sorted_eigs) / np.sum(sorted_eigs)

        return {
            "correlation_matrix": corr_matrix,
            "redundant_pairs": redundant_pairs,
            "diversifiers": diversifiers,
            "effective_n": effective_n,
            "n_strategies": len(returns_df.columns),
            "redundancy_ratio": effective_n / len(returns_df.columns),
            "variance_explained_top3": cum_var_explained[2] if len(cum_var_explained) > 2 else 1.0,
            "eigenvalues": sorted_eigs.tolist(),
        }


class RegimeConditionalTester:
    """
    Test strategy performance conditional on micro-regimes.

    For each strategy and regime:
    1. Calculate performance metrics
    2. Test if significantly different from overall
    3. Identify regime specialists
    """

    def __init__(self, min_regime_days: int = 30):
        self.min_regime_days = min_regime_days

    def test(
        self,
        strategy_returns: pd.Series,
        regime_labels: pd.Series,
        overall_sharpe: float,
    ) -> Dict:
        """
        Test regime-conditional performance.

        Returns performance by regime and significance tests.
        """
        # Align data
        aligned = pd.DataFrame({
            "returns": strategy_returns,
            "regime": regime_labels,
        }).dropna()

        results = {}
        regime_counts = aligned["regime"].value_counts()

        for regime in regime_counts.index:
            count = regime_counts[regime]
            if count < self.min_regime_days:
                continue

            regime_returns = aligned[aligned["regime"] == regime]["returns"]

            # Calculate regime-specific Sharpe
            regime_sharpe = (
                regime_returns.mean() * 252 - 0.04
            ) / (regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0

            # Test if different from overall (t-test on returns)
            other_returns = aligned[aligned["regime"] != regime]["returns"]
            if len(other_returns) > 30:
                t_stat, p_value = stats.ttest_ind(regime_returns, other_returns)
            else:
                t_stat, p_value = 0, 1.0

            results[regime] = {
                "days": count,
                "sharpe": regime_sharpe,
                "mean_return": regime_returns.mean() * 252,
                "volatility": regime_returns.std() * np.sqrt(252),
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_specialist": p_value < 0.05 and abs(regime_sharpe - overall_sharpe) > 0.5,
                "outperforms_overall": regime_sharpe > overall_sharpe,
            }

        # Identify best and worst regimes
        valid_regimes = {k: v for k, v in results.items() if v["days"] >= self.min_regime_days}
        if valid_regimes:
            best_regime = max(valid_regimes.items(), key=lambda x: x[1]["sharpe"])
            worst_regime = min(valid_regimes.items(), key=lambda x: x[1]["sharpe"])
        else:
            best_regime = worst_regime = None

        return {
            "regime_performance": results,
            "best_regime": best_regime,
            "worst_regime": worst_regime,
            "n_regimes_analyzed": len(valid_regimes),
            "regime_stability": self._calculate_stability(results) if results else 0,
        }

    def _calculate_stability(self, regime_results: Dict) -> float:
        """
        Calculate stability score across regimes.

        High stability = similar performance across regimes
        Low stability = performance varies widely by regime
        """
        sharpes = [r["sharpe"] for r in regime_results.values() if r["days"] >= self.min_regime_days]
        if len(sharpes) < 2:
            return 1.0

        # Stability = 1 - (std of Sharpes / mean of abs Sharpes)
        std_sharpe = np.std(sharpes)
        mean_abs_sharpe = np.mean(np.abs(sharpes))

        if mean_abs_sharpe == 0:
            return 0.0

        return max(0, 1 - std_sharpe / mean_abs_sharpe)


def run_full_validation(strategies: List, data: pd.DataFrame) -> pd.DataFrame:
    """
    Run complete validation framework on all strategies.

    Returns DataFrame with validation results for each strategy.
    """
    print("=" * 70)
    print("COMPREHENSIVE TA VALIDATION FRAMEWORK")
    print("=" * 70)
    print()

    # Setup
    config = BacktestConfig(
        initial_capital=500_000,
        max_leverage=2.0,
        max_drawdown=0.20,
        warmup_period=200,
    )
    engine = BacktestEngine(config=config)

    mc_validator = MonteCarloValidator(n_simulations=500)
    corr_analyzer = StrategyCorrelationAnalyzer()
    regime_tester = RegimeConditionalTester()

    # Phase 1: Individual Strategy Testing
    print("PHASE 1: Monte Carlo Validation")
    print("-" * 40)

    results = {}
    returns_dict = {}
    p_values = {}

    for strategy in strategies:
        print(f"  Testing {strategy.name}...", end=" ", flush=True)

        try:
            # Run Monte Carlo validation
            mc_result = mc_validator.validate(
                strategy, data, engine, metric="sharpe_ratio"
            )

            # Store results
            results[strategy.name] = mc_result
            p_values[strategy.name] = mc_result["p_value"]

            # Store returns for correlation analysis
            backtest_result = engine.run(strategy, data)
            returns_dict[strategy.name] = backtest_result.returns

            # Reset strategy state if needed
            if hasattr(strategy, '_last_rebalance_idx'):
                strategy._last_rebalance_idx = -999

            print(f"Sharpe={mc_result['real_metric']:.2f}, p={mc_result['p_value']:.3f}")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Phase 2: Multiple Testing Correction
    print()
    print("PHASE 2: Multiple Testing Correction")
    print("-" * 40)

    corrector = MultipleTestingCorrector()

    bonferroni_p = corrector.bonferroni(p_values)
    holm_p = corrector.holm_bonferroni(p_values)
    bh_results = corrector.benjamini_hochberg(p_values, fdr=0.05)

    print(f"  Strategies tested: {len(p_values)}")
    print(f"  Raw p < 0.05: {sum(1 for p in p_values.values() if p < 0.05)}")
    print(f"  Bonferroni p < 0.05: {sum(1 for p in bonferroni_p.values() if p < 0.05)}")
    print(f"  Holm p < 0.05: {sum(1 for p in holm_p.values() if p < 0.05)}")
    print(f"  BH FDR 5%: {sum(1 for _, sig in bh_results.values() if sig)}")

    # Phase 3: Correlation Analysis
    print()
    print("PHASE 3: Strategy Correlation Analysis")
    print("-" * 40)

    corr_results = corr_analyzer.analyze(returns_dict)

    if "error" not in corr_results:
        print(f"  Effective N strategies: {corr_results['effective_n']:.1f} / {corr_results['n_strategies']}")
        print(f"  Redundancy ratio: {corr_results['redundancy_ratio']:.1%}")
        print(f"  Variance explained by top 3: {corr_results['variance_explained_top3']:.1%}")

        if corr_results["redundant_pairs"]:
            print(f"  Redundant pairs (|r| > 0.7):")
            for s1, s2, r in corr_results["redundant_pairs"][:5]:
                print(f"    {s1} <-> {s2}: r={r:.2f}")

        if corr_results["diversifiers"]:
            print(f"  Diversifying pairs (r < -0.3):")
            for s1, s2, r in corr_results["diversifiers"][:3]:
                print(f"    {s1} <-> {s2}: r={r:.2f}")

    # Build summary DataFrame
    summary_rows = []
    for name in results:
        mc = results[name]
        bh_p, bh_sig = bh_results.get(name, (1.0, False))

        summary_rows.append({
            "strategy": name,
            "sharpe": mc["real_metric"],
            "sharpe_ci_lower": mc["ci_lower"],
            "sharpe_ci_upper": mc["ci_upper"],
            "p_value_raw": mc["p_value"],
            "p_value_bonferroni": bonferroni_p.get(name, 1.0),
            "p_value_bh": bh_p,
            "significant_bh": bh_sig,
            "percentile_vs_random": mc["percentile"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("sharpe", ascending=False)

    print()
    print("PHASE 4: Summary Results")
    print("-" * 40)
    print()
    print(summary_df.to_string(index=False))

    return summary_df, corr_results


def main():
    """Run validation framework."""
    # Import strategies
    from strategies.trend_following import (
        GoldenCrossStrategy, MACDTrendStrategy, ADXBreakoutStrategy,
        IchimokuStrategy, ParabolicSARStrategy, DonchianBreakoutStrategy
    )
    from strategies.mean_reversion import (
        RSIReversalStrategy, BollingerBounceStrategy, StochasticStrategy,
        WilliamsRStrategy, CCIReversalStrategy
    )
    from strategies.volatility import (
        ATRBreakoutStrategy, KeltnerBreakoutStrategy, BBSqueezeStrategy
    )
    from strategies.momentum import (
        MomentumStrategy, AroonTrendStrategy, TRIXTrendStrategy
    )
    from strategies.volume import (
        OBVConfirmationStrategy, MFIReversalStrategy, VWAPReversionStrategy
    )
    from strategies.base import BuyAndHoldStrategy, SMA200Strategy
    from strategies.academic_baselines import get_all_academic_baselines

    # Load data
    print("Loading QQQ data (2015-2024)...")
    loader = QQQDataLoader(start_date="2015-01-01", end_date="2024-12-31")
    data = loader.fetch()

    if data is None or len(data) < 1000:
        print("ERROR: Could not load sufficient data")
        return

    print(f"Loaded {len(data)} days of data")
    print()

    # All strategies to test
    strategies = [
        # Baselines
        BuyAndHoldStrategy(),
        SMA200Strategy(),
        # Academic baselines
        *get_all_academic_baselines(),
        # Trend following
        GoldenCrossStrategy(),
        MACDTrendStrategy(),
        ADXBreakoutStrategy(),
        IchimokuStrategy(),
        ParabolicSARStrategy(),
        DonchianBreakoutStrategy(),
        # Mean reversion
        RSIReversalStrategy(),
        BollingerBounceStrategy(),
        StochasticStrategy(),
        WilliamsRStrategy(),
        CCIReversalStrategy(),
        # Volatility
        ATRBreakoutStrategy(),
        KeltnerBreakoutStrategy(),
        BBSqueezeStrategy(),
        # Momentum
        MomentumStrategy(),
        AroonTrendStrategy(),
        TRIXTrendStrategy(),
        # Volume
        OBVConfirmationStrategy(),
        MFIReversalStrategy(),
        VWAPReversionStrategy(),
    ]

    # Run validation
    summary_df, corr_results = run_full_validation(strategies, data)

    # Save results
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/ta_validation_results.csv", index=False)
    print()
    print("Results saved to results/ta_validation_results.csv")

    # Final verdict
    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    sig_count = summary_df["significant_bh"].sum()
    total = len(summary_df)

    print(f"Strategies with statistically significant alpha: {sig_count}/{total}")
    print()

    if sig_count > 0:
        print("Significant strategies (after FDR correction):")
        sig_strategies = summary_df[summary_df["significant_bh"]]
        for _, row in sig_strategies.iterrows():
            print(f"  {row['strategy']}: Sharpe={row['sharpe']:.2f}, p={row['p_value_bh']:.4f}")
    else:
        print("⚠️  No strategies survived multiple testing correction!")
        print("   This suggests all alpha may be due to data mining / chance.")


if __name__ == "__main__":
    main()

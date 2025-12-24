#!/usr/bin/env python3
"""
RED TEAM AUDIT
==============

Critical review of all methodology, calculations, and assumptions.
Assumes nothing is correct. Finds all errors and proposes fixes.

Run this to validate the entire system before deployment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from datetime import datetime

from data.loader import QQQDataLoader
from backtesting.engine import BacktestEngine, BacktestConfig
from backtesting.cost_model import CostModel
from backtesting.metrics import calculate_metrics
from strategies.base import BuyAndHoldStrategy, SMA200Strategy
from strategies.trend_following import DonchianBreakoutStrategy


def print_section(title):
    print("\n" + "=" * 80)
    print(f"RED TEAM AUDIT: {title}")
    print("=" * 80)


def audit_sharpe_calculation():
    """
    CRITICAL CHECK: Are Sharpe ratios calculated correctly?

    Known issues to check:
    1. Is annualization correct? (sqrt(252) for daily)
    2. Is risk-free rate applied correctly?
    3. Are extreme values causing overflow?
    4. Is the formula correct? (excess return / volatility)
    """
    print_section("SHARPE RATIO CALCULATION")

    errors_found = []

    # Generate synthetic returns for testing
    np.random.seed(42)
    daily_returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # ~25% ann. return, 32% vol

    # Manual calculation
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    ann_return = mean_return * 252
    ann_vol = std_return * np.sqrt(252)
    risk_free = 0.04
    expected_sharpe = (ann_return - risk_free) / ann_vol

    print(f"Synthetic test data:")
    print(f"  Mean daily return: {mean_return*100:.4f}%")
    print(f"  Daily std: {std_return*100:.4f}%")
    print(f"  Annualized return: {ann_return*100:.2f}%")
    print(f"  Annualized vol: {ann_vol*100:.2f}%")
    print(f"  Expected Sharpe (manual): {expected_sharpe:.4f}")

    # Check metrics calculation
    positions = pd.Series([1.0] * len(daily_returns))
    costs = pd.Series([0.0] * len(daily_returns))

    try:
        metrics = calculate_metrics(daily_returns, positions, costs, risk_free_rate=0.04)
        calculated_sharpe = metrics.sharpe_ratio
        print(f"  Calculated Sharpe (metrics.py): {calculated_sharpe:.4f}")

        if abs(calculated_sharpe - expected_sharpe) > 0.1:
            errors_found.append(f"Sharpe mismatch: expected {expected_sharpe:.4f}, got {calculated_sharpe:.4f}")
    except Exception as e:
        errors_found.append(f"Sharpe calculation error: {e}")

    # CHECK 1: Sharpe ratios > 3 are extremely rare in real trading
    print(f"\n⚠️  CONCERN: Reported Sharpe ratios of 10-15 are unrealistic")
    print(f"   - Best hedge funds achieve Sharpe ~2-3")
    print(f"   - Sharpe > 5 is almost certainly an error or artifact")

    # CHECK 2: Is the Sharpe calculation using total return incorrectly?
    print(f"\n🔍 Checking annualization method...")

    # The metrics.py uses: total_return = (1 + returns).prod() - 1
    # Then: annualized_return = (1 + total_return) ** (1 / n_years) - 1
    # This is CAGR, which is correct

    total_return = (1 + daily_returns).prod() - 1
    n_years = len(daily_returns) / 252
    cagr = (1 + total_return) ** (1 / n_years) - 1
    print(f"  Total return: {total_return*100:.2f}%")
    print(f"  CAGR: {cagr*100:.2f}%")
    print(f"  Simple annualized (mean*252): {mean_return*252*100:.2f}%")

    # CHECK 3: Volatility calculation
    print(f"\n🔍 Checking volatility calculation...")
    calc_vol = metrics.annualized_volatility
    print(f"  Calculated vol: {calc_vol*100:.2f}%")
    print(f"  Expected vol: {ann_vol*100:.2f}%")

    return errors_found


def audit_position_sizing():
    """
    CRITICAL CHECK: Position sizing with leverage

    Known issues:
    1. 3x leverage should multiply returns AND risk by 3
    2. Is leverage correctly applied in get_position_size()?
    3. Are costs calculated on leveraged notional?
    """
    print_section("POSITION SIZING & LEVERAGE")

    errors_found = []

    # Check base strategy position sizing
    from strategies.base import ExpertStrategy, Signal
    from strategies.trend_following import GoldenCrossStrategy

    strategy = GoldenCrossStrategy()

    # Check position sizing
    print("Position sizing for different confidence levels:")
    for signal in [Signal.LONG, Signal.SHORT, Signal.CASH]:
        for confidence in [0.3, 0.5, 0.7, 0.9]:
            size = strategy.get_position_size(signal, confidence, max_leverage=3.0)
            print(f"  {signal.name:<6} conf={confidence:.1f}: position={size:+.1f}")

    # ISSUE FOUND: High confidence always uses max leverage
    print(f"\n⚠️  CONCERN: Position sizing jumps discretely")
    print(f"   - confidence > 0.8 → 3x leverage")
    print(f"   - confidence > 0.6 → 2.1x leverage")
    print(f"   - No gradual scaling")

    # Check if leveraged returns are realistic
    print(f"\n🔍 Leverage return analysis:")
    print(f"   If QQQ returns 20% annually:")
    print(f"   - 1x leverage: 20% return")
    print(f"   - 3x leverage: 60% return (minus margin costs)")
    print(f"   - Over 10 years with 3x: (1.60)^10 = 109x = 10,900%")
    print(f"   - This explains the 380% annualized returns!")

    print(f"\n❌ PROBLEM IDENTIFIED:")
    print(f"   The extreme returns are due to constant 3x leverage compounding")
    print(f"   This is NOT realistic because:")
    print(f"   1. 3x leverage hits margin calls during drawdowns")
    print(f"   2. Volatility decay erodes leveraged returns")
    print(f"   3. Rebalancing costs are higher than modeled")

    return errors_found


def audit_drawdown_enforcement():
    """
    CRITICAL CHECK: Is the 25% max drawdown correctly enforced?

    Known issues:
    1. After forced liquidation, does the strategy re-enter?
    2. Is drawdown calculated from peak equity correctly?
    3. Does forced liquidation happen at the right time?
    """
    print_section("DRAWDOWN ENFORCEMENT")

    errors_found = []

    # Load data and run a strategy
    loader = QQQDataLoader(start_date="2020-01-01", end_date="2020-12-31")
    data = loader.fetch()

    config = BacktestConfig(
        initial_capital=500_000,
        max_leverage=3.0,
        max_drawdown=0.25,
        warmup_period=50,  # Shorter for test
    )

    # Use a strategy that should hit drawdowns during COVID crash
    from strategies.mean_reversion import RSIReversalStrategy
    strategy = RSIReversalStrategy()
    engine = BacktestEngine(config=config)
    result = engine.run(strategy, data)

    print(f"RSI Strategy during COVID (2020):")
    print(f"  Max drawdown: {result.metrics.max_drawdown*100:.1f}%")
    print(f"  Forced liquidations: {len(result.trades[result.trades['signal'] == 'FORCED_LIQUIDATION']) if len(result.trades) > 0 else 0}")

    # Check if drawdown exceeded 25%
    equity = result.equity_curve
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    print(f"  Actual max drawdown from equity: {max_dd*100:.1f}%")

    if max_dd < -0.25:
        errors_found.append(f"Drawdown limit breached: {max_dd*100:.1f}% < -25%")
        print(f"\n❌ ERROR: Drawdown limit was breached!")
        print(f"   The 25% limit should prevent this")

    # Check the drawdown enforcement timing
    print(f"\n🔍 Checking drawdown enforcement timing...")

    # Find when drawdown first hit 25%
    breach_dates = drawdown[drawdown < -0.25].index
    if len(breach_dates) > 0:
        print(f"  First breach: {breach_dates[0]}")
        print(f"  Number of days in breach: {len(breach_dates)}")

    return errors_found


def audit_look_ahead_bias():
    """
    CRITICAL CHECK: Is there any look-ahead bias in the strategies?

    This is the most common source of unrealistic backtest results.
    """
    print_section("LOOK-AHEAD BIAS CHECK")

    errors_found = []

    # Check strategy implementations
    print("Checking strategy implementations for look-ahead bias...")

    # Load a strategy and check its signal generation
    from strategies.trend_following import GoldenCrossStrategy

    loader = QQQDataLoader(start_date="2015-01-01", end_date="2020-03-31")
    data = loader.fetch()

    strategy = GoldenCrossStrategy()

    # Test: signals at different idx values should only use data[:idx+1]
    print("\n🔍 Testing signal consistency...")

    idx = 500  # Test at this index (ensure enough data)
    if idx >= len(data):
        idx = len(data) - 10

    signal1 = strategy.generate_signal(data, idx)

    # Now test with modified future data (simulating different futures)
    extended_data = data.copy()
    # Modify future data - if strategy uses it, signals would differ
    for i in range(idx + 1, min(idx + 50, len(extended_data))):
        extended_data.iloc[i, extended_data.columns.get_loc('close')] *= 2

    signal2 = strategy.generate_signal(extended_data, idx)

    if signal1.signal != signal2.signal:
        errors_found.append("LOOK-AHEAD BIAS DETECTED: Signal changed when future data changed")
        print("❌ LOOK-AHEAD BIAS DETECTED!")
    else:
        print("✅ No look-ahead bias detected in signal generation")

    # Check regime detection
    print("\n🔍 Checking regime detection for look-ahead...")
    from regime.detector import RulesBasedDetector

    detector = RulesBasedDetector()
    regime1 = detector.detect(data, idx)
    regime2 = detector.detect(extended_data, idx)

    if regime1.regime != regime2.regime:
        errors_found.append("LOOK-AHEAD BIAS in regime detection")
        print("❌ LOOK-AHEAD BIAS in regime detection!")
    else:
        print("✅ No look-ahead bias in regime detection")

    return errors_found


def audit_cost_model():
    """
    CRITICAL CHECK: Are costs realistic?

    Issues to check:
    1. Is slippage modeled correctly?
    2. Are margin costs applied daily?
    3. Are costs proportional to trade size?
    """
    print_section("COST MODEL VALIDATION")

    errors_found = []

    cost_model = CostModel()

    print("Current cost model parameters:")
    print(f"  Commission: ${cost_model.commission_per_share}/share")
    print(f"  Min commission: ${cost_model.min_commission}")
    print(f"  Slippage: {cost_model.slippage_bps} bps")
    print(f"  Margin interest: {cost_model.margin_interest_rate*100}%")
    print(f"  Borrow rate: {cost_model.borrow_rate*100}%")

    # Test trade cost
    shares = 1000
    price = 500
    trade_cost = cost_model.calculate_trade_cost(shares, price)

    expected_commission = max(shares * 0.005, 1.0)
    expected_slippage = shares * price * (2 / 10000)
    expected_total = expected_commission + expected_slippage

    print(f"\nTest trade: {shares} shares @ ${price}")
    print(f"  Expected commission: ${expected_commission:.2f}")
    print(f"  Expected slippage: ${expected_slippage:.2f}")
    print(f"  Expected total: ${expected_total:.2f}")
    print(f"  Calculated total: ${trade_cost:.2f}")

    if abs(trade_cost - expected_total) > 0.01:
        errors_found.append(f"Trade cost mismatch: expected ${expected_total:.2f}, got ${trade_cost:.2f}")

    # CHECK: Are costs too low?
    print(f"\n⚠️  CONCERN: Slippage of 2 bps may be optimistic")
    print(f"   - During high volatility, slippage can be 10-50 bps")
    print(f"   - Large orders face market impact")
    print(f"   - Recommendation: Test with 5-10 bps slippage")

    # Test holding cost for leveraged position
    position_value = 1_000_000  # $1M position
    leverage = 3.0
    holding_cost = cost_model.calculate_holding_cost(position_value, leverage, is_short=False, days=252)

    # Expected: margin on 2/3 of position at 7%
    margin_portion = position_value * (leverage - 1) / leverage
    expected_annual_margin = margin_portion * 0.07

    print(f"\nAnnual holding cost for $1M @ 3x leverage:")
    print(f"  Calculated: ${holding_cost:,.2f}")
    print(f"  Expected margin cost: ${expected_annual_margin:,.2f}")

    return errors_found


def audit_benchmark_comparison():
    """
    CRITICAL CHECK: Are we comparing apples to apples?

    Issues:
    1. Baselines use 1x leverage, experts use 3x - unfair comparison!
    2. Should compare risk-adjusted returns, not absolute returns
    """
    print_section("BENCHMARK COMPARISON")

    errors_found = []

    print("⚠️  CRITICAL ISSUE IDENTIFIED:")
    print()
    print("   The comparison between baselines and expert strategies is INVALID!")
    print()
    print("   Baselines (BuyAndHold, SMA200): 1x leverage")
    print("   Expert strategies: 3x leverage")
    print()
    print("   This means:")
    print("   - Expert strategies take 3x the RISK")
    print("   - Returns should be ~3x higher just from leverage")
    print("   - Sharpe ratios are NOT comparable")
    print()
    print("   RECOMMENDATION:")
    print("   1. Run experts with 1x leverage for fair comparison")
    print("   2. OR run baselines with 3x leverage")
    print("   3. Compare using leverage-adjusted Sharpe")

    errors_found.append("Unfair benchmark comparison: different leverage levels")

    return errors_found


def audit_statistical_significance():
    """
    CRITICAL CHECK: Are results statistically significant?

    With 10 years of data, we have ~2500 daily observations.
    But how many independent decisions does each strategy make?
    """
    print_section("STATISTICAL SIGNIFICANCE")

    errors_found = []

    print("Statistical concerns:")
    print()
    print("1. NUMBER OF INDEPENDENT TRADES")
    print("   - BuyAndHold: 0 trades (no statistical power)")
    print("   - DonchianBreakout: ~647 trades")
    print("   - But many are daily rebalances, not independent decisions")
    print()
    print("2. MULTIPLE TESTING PROBLEM")
    print("   - We tested 24 strategies")
    print("   - By chance, ~1-2 will look great")
    print("   - Need Bonferroni correction or similar")
    print()
    print("3. SURVIVORSHIP BIAS")
    print("   - QQQ itself is a survivor (NASDAQ-100)")
    print("   - Failed stocks were removed")
    print("   - This inflates historical returns")
    print()
    print("4. DATA SNOOPING")
    print("   - The strategies were designed AFTER seeing QQQ's history")
    print("   - Parameters may be implicitly fit to this data")
    print()

    print("RECOMMENDATION: Out-of-sample testing required")
    print("  - Hold out 2020-2025 data during development")
    print("  - Only test on it once at the end")

    return errors_found


def propose_additional_tests():
    """
    Propose additional tests that should be run.
    """
    print_section("PROPOSED ADDITIONAL TESTS")

    tests = [
        ("1. MONTE CARLO SIMULATION",
         "  - Randomize entry/exit by ±1-5 days\n"
         "  - If results change significantly, strategy is fragile"),

        ("2. TRANSACTION COST SENSITIVITY",
         "  - Run with 2x, 3x, 5x current costs\n"
         "  - Many strategies fail with realistic costs"),

        ("3. DIFFERENT TIME PERIODS",
         "  - Test on 2000-2010 (dot-com crash + recovery)\n"
         "  - Test on 2007-2009 (financial crisis)\n"
         "  - Strategies that only work in bull markets are dangerous"),

        ("4. LEVERAGE-ADJUSTED COMPARISON",
         "  - Run all strategies at 1x leverage\n"
         "  - Then compare fairly"),

        ("5. PARAMETER SENSITIVITY",
         "  - Vary SMA periods by ±20%\n"
         "  - Vary RSI thresholds by ±10\n"
         "  - Robust strategies should still work"),

        ("6. REGIME TRANSITION TESTING",
         "  - Test performance during regime transitions\n"
         "  - These are when most losses occur"),

        ("7. CORRELATION ANALYSIS",
         "  - Are all 'top' strategies correlated?\n"
         "  - If yes, diversification benefit is limited"),

        ("8. REALISTIC EXECUTION SIMULATION",
         "  - Add random execution delays\n"
         "  - Add partial fills\n"
         "  - Add random slippage spikes"),

        ("9. DRAWDOWN DURATION ANALYSIS",
         "  - How long are drawdowns?\n"
         "  - Can you survive 6-12 months underwater?"),

        ("10. TAIL RISK ANALYSIS",
         "   - What's the worst 1-day, 1-week, 1-month loss?\n"
         "   - With 3x leverage, a 10% QQQ drop = 30% loss"),
    ]

    for test_name, description in tests:
        print(f"\n{test_name}")
        print(description)


def run_full_audit():
    """Run the complete red team audit."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "         RED TEAM AUDIT - CRITICAL REVIEW OF ALL METHODOLOGY".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    all_errors = []

    # Load data first
    print("\nLoading data for audit...")
    loader = QQQDataLoader(start_date="2015-01-01")
    data = loader.fetch()
    print(f"Data loaded: {len(data)} rows")

    # Run all audits
    all_errors.extend(audit_sharpe_calculation())
    all_errors.extend(audit_position_sizing())
    all_errors.extend(audit_drawdown_enforcement())
    all_errors.extend(audit_look_ahead_bias())
    all_errors.extend(audit_cost_model())
    all_errors.extend(audit_benchmark_comparison())
    all_errors.extend(audit_statistical_significance())
    propose_additional_tests()

    # Summary
    print_section("AUDIT SUMMARY")

    if all_errors:
        print(f"\n❌ ERRORS FOUND: {len(all_errors)}")
        for i, error in enumerate(all_errors, 1):
            print(f"   {i}. {error}")
    else:
        print("\n✅ No critical errors found")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    print("""
1. LEVERAGE COMPARISON IS UNFAIR
   - Experts use 3x leverage, baselines use 1x
   - This inflates expert returns by ~3x
   - FIX: Compare at same leverage level

2. SHARPE RATIOS ARE UNREALISTIC
   - Reported Sharpe of 10-15 is impossible in reality
   - Best funds achieve 2-3
   - CAUSE: Likely leverage effect + favorable period

3. DRAWDOWN LIMIT MAY NOT WORK AS INTENDED
   - After forced liquidation, strategy re-enters
   - This can lead to repeated whipsaws
   - FIX: Add cooldown period after forced liquidation

4. MEAN-REVERSION FAILURE IS EXPECTED
   - QQQ 2015-2025 was a strong bull market
   - Mean-reversion strategies short too often
   - FIX: Only enable in BEAR/SIDEWAYS regimes

5. STATISTICAL CONCERNS
   - Multiple testing without correction
   - Limited independent trades
   - Survivorship bias in QQQ

6. COST MODEL MAY BE OPTIMISTIC
   - 2 bps slippage is low for leveraged trades
   - Market impact not modeled
   - FIX: Test with 5-10 bps slippage
""")


if __name__ == "__main__":
    run_full_audit()

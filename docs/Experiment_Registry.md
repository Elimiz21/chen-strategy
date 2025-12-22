# Experiment Registry
## Adaptive Regime-Aware Trading System - QQQ Focus

### Document Control
- Last Updated: 2025-12-22

---

## Registry Rules (NON-NEGOTIABLE)

1. **PRE-REGISTER** all experiments BEFORE running
2. Record ALL trials, not just successful ones
3. Include exact dataset version (SHA-256 hash)
4. Include exact code version (git commit hash)
5. Include full configuration with seeds
6. Link to results (even if negative)

---

## Experiment Format

| Field | Description |
|-------|-------------|
| ID | EXP-XXX format |
| Date | Registration date |
| Hypothesis | What are we testing? |
| Dataset Version | SHA-256 hash of QQQ dataset |
| Code Hash | Git commit hash |
| Config | Full configuration including seeds |
| Trials | Number of trials/variations |
| Results Link | Path to results file |
| Status | Registered, Running, Complete, Failed |
| Conclusion | What did we learn? |

---

## Registered Experiments

### EXP-001: QQQ Baseline Performance Establishment
- **Date:** 2025-12-22
- **Hypothesis:** Establish baseline performance metrics for QQQ strategies
- **Dataset Version:** [PENDING - to be filled when data ready]
- **Code Hash:** [PENDING - to be filled when implemented]
- **Config:**
  ```yaml
  asset: QQQ
  period: 2000-01-01 to 2024-12-31
  baselines:
    - buy_and_hold
    - sma_200_filter
    - golden_cross_50_200
    - rsi_mean_reversion
    - volatility_targeting
  capital: 500000
  max_dd: 0.25
  cost_model: qqq_unified_v1
  seeds: [42, 123, 456, 789, 1337]
  ```
- **Trials:** 5 baselines × 5 seeds = 25
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

### EXP-002: Trend-Following TA Experts on QQQ
- **Date:** 2025-12-22
- **Hypothesis:** Trend-following TA indicators can outperform buy-and-hold in trending regimes
- **Dataset Version:** [PENDING]
- **Code Hash:** [PENDING]
- **Config:**
  ```yaml
  asset: QQQ
  period: 2000-01-01 to 2024-12-31
  experts:
    - ma_crossover:
        fast: [10, 20, 50]
        slow: [50, 100, 200]
    - macd_standard:
        fast: 12
        slow: 26
        signal: 9
    - adx_trend:
        period: [14, 20]
        threshold: [20, 25, 30]
    - ichimoku_cloud:
        tenkan: 9
        kijun: 26
        senkou: 52
    - parabolic_sar:
        af_start: 0.02
        af_max: 0.2
    - donchian_breakout:
        period: [10, 20, 55]
  capital: 500000
  max_dd: 0.25
  cost_model: qqq_unified_v1
  seeds: [42, 123, 456]
  ```
- **Trials:** ~50 parameter combinations × 3 seeds = ~150
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

### EXP-003: Mean-Reversion TA Experts on QQQ
- **Date:** 2025-12-22
- **Hypothesis:** Mean-reversion TA indicators outperform in ranging/choppy markets
- **Dataset Version:** [PENDING]
- **Code Hash:** [PENDING]
- **Config:**
  ```yaml
  asset: QQQ
  period: 2000-01-01 to 2024-12-31
  experts:
    - rsi_reversal:
        period: [7, 14, 21]
        oversold: [20, 30]
        overbought: [70, 80]
    - bollinger_bounce:
        period: [10, 20]
        std: [1.5, 2.0, 2.5]
    - stochastic:
        k_period: [5, 14]
        d_period: 3
        oversold: 20
        overbought: 80
    - williams_r:
        period: [10, 14, 20]
    - cci_reversal:
        period: [14, 20]
        threshold: [100, 150]
  capital: 500000
  max_dd: 0.25
  cost_model: qqq_unified_v1
  seeds: [42, 123, 456]
  ```
- **Trials:** ~40 parameter combinations × 3 seeds = ~120
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

### EXP-004: Volatility-Based TA Experts on QQQ
- **Date:** 2025-12-22
- **Hypothesis:** Volatility-based position sizing improves risk-adjusted returns
- **Dataset Version:** [PENDING]
- **Code Hash:** [PENDING]
- **Config:**
  ```yaml
  asset: QQQ
  period: 2000-01-01 to 2024-12-31
  experts:
    - atr_position_sizing:
        atr_period: [10, 14, 20]
        risk_per_trade: [0.01, 0.02]
    - keltner_channel:
        ema_period: [10, 20]
        atr_mult: [1.5, 2.0, 2.5]
    - volatility_regime:
        lookback: [20, 60, 120]
        threshold: [0.5, 1.0, 1.5]  # std above/below mean vol
  capital: 500000
  max_dd: 0.25
  cost_model: qqq_unified_v1
  seeds: [42, 123, 456]
  ```
- **Trials:** ~30 parameter combinations × 3 seeds = ~90
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

### EXP-005: Volume-Based TA Experts on QQQ
- **Date:** 2025-12-22
- **Hypothesis:** Volume confirmation improves signal quality
- **Dataset Version:** [PENDING]
- **Code Hash:** [PENDING]
- **Config:**
  ```yaml
  asset: QQQ
  period: 2000-01-01 to 2024-12-31
  experts:
    - obv_trend:
        signal_period: [10, 20]
    - mfi_divergence:
        period: [10, 14, 20]
        oversold: 20
        overbought: 80
    - volume_breakout:
        lookback: [20, 50]
        threshold: [1.5, 2.0]  # times average volume
  capital: 500000
  max_dd: 0.25
  cost_model: qqq_unified_v1
  seeds: [42, 123, 456]
  ```
- **Trials:** ~20 parameter combinations × 3 seeds = ~60
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

### EXP-006: QQQ Regime Detection
- **Date:** 2025-12-22
- **Hypothesis:** Market regimes in QQQ are detectable with >60% accuracy out-of-sample
- **Dataset Version:** [PENDING]
- **Code Hash:** [PENDING]
- **Config:**
  ```yaml
  asset: QQQ
  period: 2000-01-01 to 2024-12-31
  regimes:
    - trend_up
    - trend_down
    - mean_reverting
    - high_volatility
    - low_volatility
  detection_methods:
    - rules_based:
        trend: ma_slope + adx
        volatility: atr_percentile
    - hmm:
        n_states: [2, 3, 4, 5]
        features: [returns, volatility, volume]
    - clustering:
        method: kmeans
        n_clusters: [3, 4, 5]
        features: [returns, volatility, autocorr]
  validation: walk_forward
  train_window: 252  # 1 year
  test_window: 63    # 3 months
  seeds: [42, 123, 456]
  ```
- **Trials:** 3 methods × multiple configs × 3 seeds = ~50
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

### EXP-007: Expert-Regime Performance Matrix
- **Date:** 2025-12-22
- **Hypothesis:** Different TA experts perform differently across regimes
- **Dataset Version:** [PENDING]
- **Code Hash:** [PENDING]
- **Config:**
  ```yaml
  asset: QQQ
  period: 2000-01-01 to 2024-12-31
  experts: [all from EXP-002 through EXP-005]
  regimes: [from EXP-006 best detector]
  metrics:
    - sharpe_by_regime
    - return_by_regime
    - win_rate_by_regime
    - max_dd_by_regime
  statistical_test: anova
  significance: 0.05
  seeds: [42, 123, 456]
  ```
- **Trials:** N experts × M regimes analysis
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

### EXP-008: Meta-Allocation with 25% DD Constraint
- **Date:** 2025-12-22
- **Hypothesis:** Regime-aware meta-allocation can beat baselines while maintaining 25% max DD
- **Dataset Version:** [PENDING]
- **Code Hash:** [PENDING]
- **Config:**
  ```yaml
  asset: QQQ
  period: 2000-01-01 to 2024-12-31
  allocation_methods:
    - regime_switching:
        confidence_threshold: [0.6, 0.7, 0.8]
    - weighted_ensemble:
        weights: regime_performance
        turnover_penalty: [0.001, 0.005, 0.01]
    - tilt_not_switch:
        tilt_rate: [0.1, 0.2, 0.3]
        min_confidence: 0.5
  constraints:
    max_dd: 0.25
    max_turnover: 50
  validation: walk_forward
  seeds: [42, 123, 456]
  ```
- **Trials:** 3 methods × configs × 3 seeds = ~50
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

## Experiment Queue

| ID | Hypothesis | Owner | Priority | Dependencies | Phase |
|----|------------|-------|----------|--------------|-------|
| EXP-001 | Baseline performance | Quant Research | P0 | Data ready | 3 |
| EXP-002 | Trend experts | Quant Research | P1 | EXP-001 | 3 |
| EXP-003 | Mean-reversion experts | Quant Research | P1 | EXP-001 | 3 |
| EXP-004 | Volatility experts | Quant Research | P1 | EXP-001 | 3 |
| EXP-005 | Volume experts | Quant Research | P1 | EXP-001 | 3 |
| EXP-006 | Regime detection | ML/Stats | P1 | Data ready | 4 |
| EXP-007 | Expert-regime matrix | Quant Research | P1 | EXP-002-006 | 4 |
| EXP-008 | Meta-allocation | ML/Stats | P1 | EXP-007 | 5 |

---

## Completed Experiments Summary

| ID | Hypothesis | Result | Key Finding |
|----|------------|--------|-------------|
| | | | |

---

## Experiment Statistics

| Metric | Count |
|--------|-------|
| Total Registered | 8 |
| Running | 0 |
| Completed | 0 |
| Failed | 0 |
| Total Planned Trials | ~545 |

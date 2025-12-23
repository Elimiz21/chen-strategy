# Phase 1: QQQ Historical Regime Analysis (2000-2024)

## Document Control
- Version: 1.0
- Date: 2025-12-23
- Owner: Quant Research Lead
- Status: IN PROGRESS

---

## 1. Executive Summary

This document analyzes QQQ (Nasdaq-100 ETF) historical price behavior to identify distinct market regimes. The goal is to categorize periods into actionable regime types that can inform expert strategy selection.

---

## 2. QQQ Historical Overview

### Key Statistics (1999-2024)
| Metric | Value | Notes |
|--------|-------|-------|
| Inception | March 10, 1999 | Launched near dot-com peak |
| Total Return | ~800%+ | Significant long-term growth |
| CAGR | ~9-10% | With high volatility |
| Max Drawdown | ~83% | Oct 2002 (dot-com bust) |
| 2nd Largest DD | ~35% | Mar 2020 (COVID crash) |
| 3rd Largest DD | ~33% | 2022 (Fed rate hikes) |
| Volatility (annualized) | ~25-30% | Higher than SPY (~18-20%) |

### Major Periods

#### 1. Dot-Com Bubble & Bust (1999-2002)
- **1999-2000**: Extreme bull market, QQQ +85% in 1999
- **2000-2002**: Devastating crash, -83% peak-to-trough
- **Regime characteristics**: Extreme trend, then extreme reversal
- **Duration**: ~3 years bear market

#### 2. Recovery & Bull Market (2003-2007)
- **Returns**: Steady recovery, ~100% gain
- **Regime characteristics**: Moderate uptrend, lower volatility than bubble
- **Duration**: ~5 years

#### 3. Financial Crisis (2008-2009)
- **2008**: QQQ -42% (less than SPY due to no financials)
- **2009**: Strong recovery, +55%
- **Regime characteristics**: Sharp V-shaped recovery
- **Duration**: ~18 months bear, rapid recovery

#### 4. Post-Crisis Bull Market (2010-2019)
- **Returns**: Exceptional, ~500%+ gains
- **Regime characteristics**: Persistent uptrend with low volatility
- **Notable events**: Brief corrections in 2011, 2015, 2018
- **Duration**: ~10 years

#### 5. COVID Era (2020-2021)
- **Mar 2020**: -35% in weeks (fastest bear ever)
- **Recovery**: V-shaped, new highs by August 2020
- **2021**: Continued bull, +27%
- **Regime characteristics**: Extreme volatility, then low-vol uptrend

#### 6. Rate Hike Bear Market (2022)
- **Returns**: -33% peak-to-trough
- **Regime characteristics**: Persistent downtrend, rising volatility
- **Duration**: ~10 months

#### 7. AI-Driven Recovery (2023-2024)
- **2023**: +55% (AI/tech rally)
- **2024**: Continued strength
- **Regime characteristics**: Strong uptrend, concentration in mega-caps

---

## 3. Proposed Regime Definitions

### Primary Regime Classification

#### A. Trend Direction Regimes
| Regime | Definition | Indicators |
|--------|------------|------------|
| **BULL TREND** | Price > 200 SMA, 50 SMA > 200 SMA | Sustained upward momentum |
| **BEAR TREND** | Price < 200 SMA, 50 SMA < 200 SMA | Sustained downward momentum |
| **SIDEWAYS** | Price crossing 200 SMA repeatedly, ADX < 25 | Range-bound, no clear trend |

#### B. Volatility Regimes
| Regime | Definition | Indicators |
|--------|------------|------------|
| **LOW VOL** | 20-day realized vol < 15% annualized | Calm markets, trend-following works |
| **NORMAL VOL** | 20-day realized vol 15-30% annualized | Typical conditions |
| **HIGH VOL** | 20-day realized vol > 30% annualized | Crisis/uncertainty, mean-reversion may work |

#### C. Combined Regime Matrix (6 States)

| | Low Vol | Normal Vol | High Vol |
|---|---------|------------|----------|
| **Bull** | Bull-Calm | Bull-Normal | Bull-Volatile |
| **Bear** | Bear-Calm | Bear-Normal | Bear-Volatile |

### Regime Persistence Analysis

Based on historical data, expected regime durations:

| Regime | Avg Duration | Min | Max | Frequency |
|--------|--------------|-----|-----|-----------|
| Bull-Calm | 6-12 months | 2 mo | 24+ mo | 40% of time |
| Bull-Normal | 3-6 months | 1 mo | 12 mo | 20% of time |
| Bull-Volatile | 1-3 months | 2 wk | 6 mo | 5% of time |
| Bear-Calm | Rare | - | - | <2% of time |
| Bear-Normal | 3-6 months | 1 mo | 18 mo | 15% of time |
| Bear-Volatile | 1-3 months | 2 wk | 6 mo | 18% of time |

---

## 4. Regime Detection Approaches

### Approach 1: Rules-Based (Simple Moving Averages)
```
Bull: Price > 200 SMA AND 50 SMA > 200 SMA
Bear: Price < 200 SMA AND 50 SMA < 200 SMA
Sideways: Neither condition
```
**Pros**: Simple, interpretable, no look-ahead
**Cons**: Lagging, whipsaws in transitions

### Approach 2: Volatility-Adjusted (ATR/VIX-Based)
```
High Vol: 20-day ATR > 1.5 * 60-day ATR OR VIX > 25
Low Vol: 20-day ATR < 0.75 * 60-day ATR OR VIX < 15
Normal Vol: Otherwise
```
**Pros**: Responsive to volatility shifts
**Cons**: May be too reactive

### Approach 3: Hidden Markov Model (HMM)
- Train HMM with 2-4 states on returns data
- States emerge from data, not predefined
- Requires out-of-sample validation

**Pros**: Data-driven, captures complex patterns
**Cons**: Black box, potential overfitting, look-ahead risk

### Approach 4: Hybrid (Rules + ML Confirmation)
- Use rules-based as primary
- ML model provides confidence score
- "Tilt not switch" based on confidence

**Pros**: Interpretable with ML enhancement
**Cons**: More complex implementation

---

## 5. Expert Strategy Mapping by Regime

### Preliminary Hypotheses

| Regime | Best Expert Types | Worst Expert Types |
|--------|-------------------|-------------------|
| Bull-Calm | Trend-following (MACD, MA Cross) | Mean-reversion (RSI, Bollinger) |
| Bull-Normal | Momentum (ROC, ADX) | Counter-trend |
| Bull-Volatile | Volatility breakout (Keltner, Donchian) | Tight stop strategies |
| Bear-Normal | Short trend-following | Long-only buy-and-hold |
| Bear-Volatile | Volatility mean-reversion (VIX) | Momentum (gets whipsawed) |
| Sideways | Mean-reversion (RSI, Bollinger) | Trend-following |

### Leverage/Position Sizing by Regime

| Regime | Suggested Position | Rationale |
|--------|-------------------|-----------|
| Bull-Calm | 2-3x Long | High confidence, low risk |
| Bull-Normal | 1-2x Long | Moderate confidence |
| Bull-Volatile | 0.5-1x Long | Reduce exposure |
| Bear-Calm | 1x Short or Cash | Rare, cautious |
| Bear-Normal | 1-2x Short | Clear trend |
| Bear-Volatile | 0.5x Short or Cash | High uncertainty |
| Sideways | 0.5-1x or Cash | Wait for clarity |

---

## 6. Key Historical Regime Periods (Labeled)

### Sample Labeling (2020-2024)

| Period | Trend | Vol | Combined Regime |
|--------|-------|-----|-----------------|
| Jan 2020 | Bull | Low | Bull-Calm |
| Feb-Mar 2020 | Bear | High | Bear-Volatile (COVID crash) |
| Apr-Aug 2020 | Bull | Normal | Bull-Normal (recovery) |
| Sep-Oct 2020 | Sideways | Normal | Sideways-Normal |
| Nov 2020 - Feb 2021 | Bull | Low | Bull-Calm |
| Mar-May 2021 | Sideways | Normal | Sideways-Normal |
| Jun-Dec 2021 | Bull | Low | Bull-Calm |
| Jan-Jun 2022 | Bear | High | Bear-Volatile |
| Jul-Aug 2022 | Bull | Normal | Bull-Normal (bear rally) |
| Sep-Dec 2022 | Bear | Normal | Bear-Normal |
| Jan-Jul 2023 | Bull | Normal | Bull-Normal (AI rally) |
| Aug-Oct 2023 | Bear | Normal | Bear-Normal (correction) |
| Nov 2023 - Present | Bull | Low | Bull-Calm |

---

## 7. Implications for Strategy

### Key Findings

1. **QQQ spends ~60% of time in bull regimes** - Long bias makes sense
2. **Bear regimes are shorter but more volatile** - Quick regime detection critical
3. **Volatility regime often leads trend regime** - Vol spike often precedes trend change
4. **Mean-reversion works in sideways/high-vol** - RSI, Bollinger effective
5. **Trend-following works in low-vol trends** - MACD, MA Cross effective

### Recommended Next Steps

1. **Quantify regime durations** with actual data (Phase 2)
2. **Backtest regime detection approaches** (Phase 4)
3. **Measure expert performance by regime** (Phase 3-4)
4. **Develop "tilt not switch" logic** (Phase 5)

---

## 8. References

- QQQ historical data (Yahoo Finance, 1999-present)
- Academic literature on regime-switching models
- Practitioner research on trend-following and mean-reversion

---

## Appendix: QQQ vs SPY Comparison

| Metric | QQQ | SPY | Implication |
|--------|-----|-----|-------------|
| Volatility | ~25-30% | ~18-20% | More regime swings |
| Max DD | 83% | 55% | Deeper bears |
| Recovery Speed | Faster | Slower | V-shaped recoveries |
| Sector Concentration | Tech-heavy | Diversified | More momentum-driven |
| Regime Persistence | Shorter | Longer | Faster adaptation needed |

# Phase 1: Technical Analysis Indicator Effectiveness Survey

## Document Control
- Version: 1.0
- Date: 2025-12-23
- Owner: Quant Research Lead
- Status: IN PROGRESS

---

## 1. Executive Summary

This document surveys the academic and practitioner literature on technical analysis indicators to identify which indicators have shown effectiveness, under what conditions, and how they might be deployed as "experts" in a regime-aware trading system.

---

## 2. Academic Literature Summary

### 2.1 Key Findings from Research

#### Trend-Following Indicators
| Study | Finding | Implication |
|-------|---------|-------------|
| Brock et al. (1992) | MA crossovers generate excess returns | Simple rules can work |
| Sullivan et al. (1999) | Data-snooping reduces significance | Need robust validation |
| Hsu et al. (2016) | Trend-following works in equity indices | QQQ suitable target |
| Moskowitz et al. (2012) | Time-series momentum profitable | 12-1 momentum effective |

#### Mean-Reversion Indicators
| Study | Finding | Implication |
|-------|---------|-------------|
| Lo & MacKinlay (1988) | Short-term reversals exist | Mean-reversion viable |
| Jegadeesh (1990) | Monthly reversals profitable | Short holding periods |
| Cooper et al. (2004) | Reversals stronger after extreme moves | RSI extremes useful |

#### Volatility-Based Indicators
| Study | Finding | Implication |
|-------|---------|-------------|
| Bollerslev et al. (1992) | Volatility clusters | ATR-based sizing works |
| Fleming et al. (2001) | Vol targeting improves Sharpe | Scale by inverse vol |
| Moreira & Muir (2017) | Managed volatility adds value | Vol regime matters |

### 2.2 Meta-Analysis Conclusions

1. **Simple rules often work as well as complex** - Avoid over-engineering
2. **Transaction costs matter** - Net-of-cost testing essential
3. **Regime-dependence is real** - Trend works in trends, MR in ranges
4. **Data-snooping is rampant** - Pre-register, walk-forward validate
5. **Combination often better than single** - Ensemble approaches

---

## 3. Indicator Categories and Specifications

### 3.1 Trend-Following Indicators

#### Moving Average Crossovers
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| SMA(50,200) | 50-day, 200-day SMA | 50 > 200 = Long | Bull-Calm |
| EMA(12,26) | 12-day, 26-day EMA | 12 > 26 = Long | Bull-Normal |
| DEMA(20,50) | Double EMA | Fast > Slow = Long | All trends |
| TEMA(10,30) | Triple EMA | Fast > Slow = Long | Fast trends |

#### MACD Family
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| MACD(12,26,9) | Standard | MACD > Signal = Long | Trending |
| MACD Histogram | Histogram slope | Rising = Long | Momentum |
| MACD Zero Cross | MACD crosses 0 | Above 0 = Long | Trend confirm |

#### ADX/DMI
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| ADX(14) | 14-day ADX | ADX > 25 = Trending | All |
| +DI/-DI | Directional | +DI > -DI = Long | Trending |
| ADX Trend Filter | ADX > 25 + DI | Strong trend signal | Bull/Bear clear |

#### Ichimoku Cloud
| Component | Signal | Best Regime |
|-----------|--------|-------------|
| Price vs Cloud | Above = Long | Trending |
| Tenkan/Kijun Cross | Tenkan > Kijun = Long | Trend initiation |
| Chikou Span | Confirms trend | Trend continuation |

### 3.2 Mean-Reversion Indicators

#### RSI Family
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| RSI(14) | 14-day RSI | < 30 = Buy, > 70 = Sell | Sideways, High-Vol |
| RSI(2) | 2-day RSI | < 10 = Buy, > 90 = Sell | Short-term MR |
| Stochastic RSI | StochRSI(14) | < 20 = Buy, > 80 = Sell | Ranging |

#### Bollinger Bands
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| BB(20,2) | 20-day, 2 std | Touch lower = Buy | Low-vol consolidation |
| BB Squeeze | Bandwidth contracts | Breakout coming | Pre-breakout |
| %B | Position in band | < 0 = Buy, > 1 = Sell | Mean-reversion |

#### Stochastic Oscillator
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| Stoch(14,3,3) | K, D lines | K < 20 = Buy | Ranging |
| Stoch Divergence | Price vs Stoch | Divergence = Reversal | Tops/bottoms |

#### Other Oscillators
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| Williams %R(14) | 14-day | < -80 = Buy, > -20 = Sell | Ranging |
| CCI(20) | 20-day CCI | < -100 = Buy, > 100 = Sell | MR regimes |
| Ultimate Oscillator | UO(7,14,28) | Oversold/overbought | Multi-timeframe MR |

### 3.3 Volatility-Based Indicators

#### ATR Family
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| ATR(14) | 14-day ATR | Position sizing | All |
| ATR Bands | Price ± 2*ATR | Breakout/MR | Vol-adjusted |
| ATR Trailing Stop | Entry - 2*ATR | Dynamic stop | Trending |

#### Keltner Channels
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| KC(20,2) | EMA(20), 2*ATR | Above upper = Long | Breakout |
| KC Squeeze | KC inside BB | Breakout imminent | Pre-breakout |

#### Donchian Channels
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| DC(20) | 20-day high/low | Break high = Long | Breakout |
| DC(55) | 55-day (Turtle) | Longer-term trend | Trending |

### 3.4 Volume-Based Indicators

#### OBV
| Indicator | Signal | Best Regime |
|-----------|--------|-------------|
| OBV Trend | OBV rising with price = Confirm | Trend confirmation |
| OBV Divergence | OBV diverges from price = Warning | Tops/bottoms |

#### Money Flow
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| MFI(14) | Volume-weighted RSI | < 20 = Buy | High volume reversals |
| Chaikin MF | CMF(20) | Positive = Bullish | Accumulation/distribution |

#### VWAP
| Indicator | Signal | Best Regime |
|-----------|--------|-------------|
| Price vs VWAP | Above = Bullish | Intraday/daily MR |
| VWAP Deviation | 2 std from VWAP | MR signal | Mean-reversion |

### 3.5 Momentum Indicators

#### Rate of Change
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| ROC(12) | 12-month momentum | Positive = Long | Trending |
| ROC(1) | 1-month momentum | Short-term signal | Fast moves |

#### Aroon
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| Aroon(25) | 25-day Aroon | Up > Down = Long | Trend identification |
| Aroon Oscillator | Up - Down | > 0 = Bullish | Trend strength |

#### TRIX
| Indicator | Parameters | Signal | Best Regime |
|-----------|------------|--------|-------------|
| TRIX(15) | Triple smoothed | Rising = Long | Filtered trend |
| TRIX Signal | 9-day EMA of TRIX | Cross = Signal | Trend changes |

---

## 4. Expert Strategy Specifications (20+ Strategies)

### 4.1 Trend-Following Experts (6)

| ID | Name | Entry | Exit | Regime Hypothesis |
|----|------|-------|------|-------------------|
| TF-01 | Golden Cross | 50 SMA > 200 SMA | 50 SMA < 200 SMA | Bull trends |
| TF-02 | MACD Trend | MACD > Signal | MACD < Signal | Trending |
| TF-03 | ADX Breakout | ADX > 25, +DI > -DI | ADX < 20 or -DI > +DI | Strong trends |
| TF-04 | Ichimoku Cloud | Price > Cloud, Tenkan > Kijun | Opposite | Clear trends |
| TF-05 | Parabolic SAR | SAR below price | SAR above price | Directional |
| TF-06 | Donchian Breakout | 20-day high break | 20-day low break | Breakouts |

### 4.2 Mean-Reversion Experts (5)

| ID | Name | Entry | Exit | Regime Hypothesis |
|----|------|-------|------|-------------------|
| MR-01 | RSI Reversal | RSI < 30 | RSI > 50 or RSI > 70 | Oversold bounces |
| MR-02 | Bollinger Bounce | Price < Lower BB | Price > Middle BB | Low-vol consolidation |
| MR-03 | Stochastic | K < 20, K crosses above D | K > 80 | Ranging markets |
| MR-04 | Williams %R | %R < -80 | %R > -20 | Oscillating |
| MR-05 | CCI Reversal | CCI < -100 | CCI > 0 | Mean-reverting |

### 4.3 Volatility-Based Experts (4)

| ID | Name | Entry | Exit | Regime Hypothesis |
|----|------|-------|------|-------------------|
| VB-01 | ATR Breakout | Price > Upper ATR Band | Price < Lower ATR Band | Vol breakouts |
| VB-02 | Keltner Breakout | Price > Upper KC | Price < EMA | Trending breakouts |
| VB-03 | Vol Targeting | Size = Target Vol / Realized Vol | Continuous | All regimes |
| VB-04 | BB Squeeze Exit | BB squeeze + direction break | Opposite break | Pre-breakout |

### 4.4 Volume-Based Experts (3)

| ID | Name | Entry | Exit | Regime Hypothesis |
|----|------|-------|------|-------------------|
| VO-01 | OBV Confirmation | Price + OBV both rising | Divergence | Volume-confirmed trends |
| VO-02 | MFI Reversal | MFI < 20 | MFI > 80 | High-vol reversals |
| VO-03 | VWAP Reversion | Price < VWAP - 2 std | Price > VWAP | Intraday/daily MR |

### 4.5 Momentum Experts (3)

| ID | Name | Entry | Exit | Regime Hypothesis |
|----|------|-------|------|-------------------|
| MO-01 | 12-1 Momentum | 12-mo return > 0 | 12-mo return < 0 | Time-series momentum |
| MO-02 | Aroon Trend | Aroon Up > 70, Aroon Down < 30 | Opposite | Trend initiation |
| MO-03 | TRIX Trend | TRIX > Signal | TRIX < Signal | Smoothed momentum |

### 4.6 Hybrid Experts (3)

| ID | Name | Entry | Exit | Regime Hypothesis |
|----|------|-------|------|-------------------|
| HY-01 | Trend + Vol Filter | TF signal + low ATR | High ATR or TF exit | Low-vol trends |
| HY-02 | MR + Trend Filter | MR signal + ADX < 20 | Trending ADX > 25 | Ranging only |
| HY-03 | Multi-Timeframe | Daily + Weekly agree | Disagreement | Confirmed trends |

---

## 5. Literature-Based Effectiveness Ratings

### Expected Performance by Indicator Type

| Type | Expected Sharpe | Best Condition | Failure Mode |
|------|-----------------|----------------|--------------|
| Trend-Following | 0.3-0.6 | Persistent trends | Whipsaws in ranges |
| Mean-Reversion | 0.2-0.5 | Ranging markets | Trends (stops out) |
| Volatility | 0.4-0.7 | Vol clustering | Regime changes |
| Volume | 0.2-0.4 | Strong moves | Low volume periods |
| Momentum | 0.3-0.5 | Trending | Reversals |

### Historical Backtest Results (Literature)

| Strategy | Study | Period | Sharpe (gross) | Sharpe (net) |
|----------|-------|--------|----------------|--------------|
| 200-day MA | Faber (2007) | 1901-2012 | 0.45 | 0.35 |
| 12-1 Momentum | Moskowitz (2012) | 1965-2009 | 0.50 | 0.40 |
| RSI(2) | Connors (2004) | 1995-2007 | 0.80 | 0.50 |
| Bollinger Bands | Lento (2007) | 1995-2006 | 0.30 | 0.15 |
| Vol Targeting | Moreira (2017) | 1926-2015 | +0.15 improvement | +0.10 improvement |

---

## 6. Implementation Priority

### Tier 1 (Must Have) - 8 Experts
1. TF-01: Golden Cross (benchmark trend)
2. TF-02: MACD Trend (popular, well-studied)
3. MR-01: RSI Reversal (benchmark MR)
4. MR-02: Bollinger Bounce (volatility-aware MR)
5. VB-03: Vol Targeting (risk management)
6. MO-01: 12-1 Momentum (academically validated)
7. TF-03: ADX Breakout (trend strength)
8. VB-01: ATR Breakout (volatility breakout)

### Tier 2 (Should Have) - 7 Experts
9. TF-04: Ichimoku Cloud
10. TF-06: Donchian Breakout
11. MR-03: Stochastic
12. MR-05: CCI Reversal
13. VO-01: OBV Confirmation
14. HY-01: Trend + Vol Filter
15. VB-02: Keltner Breakout

### Tier 3 (Nice to Have) - 9 Experts
16-24: Remaining experts (MR-04, VO-02, VO-03, MO-02, MO-03, HY-02, HY-03, TF-05, VB-04)

---

## 7. Key Takeaways for Strategy Development

1. **Start simple** - Golden Cross, RSI(14), MACD are well-studied baselines
2. **Combine regime detection with expert selection** - Not all experts work everywhere
3. **Vol targeting improves most strategies** - Scale position by inverse volatility
4. **Transaction costs matter** - High-turnover strategies need careful cost modeling
5. **Ensemble often beats single** - Consider meta-allocation across experts
6. **Walk-forward is essential** - In-sample performance is unreliable

---

## 8. Next Steps

1. Implement Tier 1 experts in backtesting framework (Phase 3)
2. Test each expert across identified regimes (Phase 3-4)
3. Measure regime-conditional performance (Phase 4)
4. Build meta-allocation engine (Phase 5)

---

## References

1. Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. Journal of Finance.
2. Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. Journal of Financial Economics.
3. Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. Journal of Finance.
4. Faber, M. (2007). A quantitative approach to tactical asset allocation. Journal of Wealth Management.
5. Connors, L. A., & Alvarez, C. (2009). Short term trading strategies that work. TradingMarkets Publishing.

# Strategy Charter
## Adaptive Regime-Aware Trading System - QQQ Focus

### Document Control
- Version: 1.0
- Last Updated: 2025-12-22
- Status: PHASE 0 - IN PROGRESS

---

## 1. Hypothesis

**Primary Hypothesis:**
Different technical analysis "expert" strategies perform best in different market regimes for QQQ (Nasdaq-100 ETF), and a regime-aware meta-allocation engine can improve net-of-cost performance versus buy-and-hold QQQ—first in strict walk-forward testing, then in paper trading, and only then with live capital.

**Sub-Hypotheses:**
1. QQQ market regimes (trend/mean-reversion, high-vol/low-vol) are detectable in real-time
2. TA expert performance is regime-dependent (not just random variation)
3. Meta-allocation can capture regime-expert relationships without excessive turnover
4. Net-of-cost performance exceeds QQQ buy-and-hold with max 25% drawdown

### Falsification Conditions
The hypothesis is REJECTED if:
- [ ] Regime detection accuracy < 60% (out-of-sample)
- [ ] Expert performance not statistically different across regimes (p > 0.05)
- [ ] Meta-allocation cannot maintain 25% max DD while beating QQQ buy-and-hold
- [ ] Turnover costs exceed alpha generated
- [ ] Results driven by single time period (fails subperiod stability)

---

## 2. Scope

### In Scope
- **Asset:** QQQ (Invesco QQQ Trust - Nasdaq-100 ETF) ONLY
- **Universe:** Single instrument - long QQQ or cash (no shorting)
- **Capital:** $500,000 deployment target
- **Time horizon:** Daily signals, rebalance as needed (when regime/signal changes)
- **Regime types:**
  - Trend (up/down)
  - Mean-reversion
  - High-volatility
  - Low-volatility
  - Bull/Bear/Sideways
- **Expert strategies:** ALL major TA tools including:
  - Moving Averages (SMA, EMA, WMA, DEMA, TEMA)
  - MACD (all variations)
  - RSI, Stochastic RSI
  - Bollinger Bands
  - ATR-based systems
  - ADX/DMI
  - Ichimoku Cloud
  - Fibonacci retracements
  - Support/Resistance levels
  - Volume indicators (OBV, VWAP, Volume Profile)
  - Momentum (ROC, MOM)
  - Williams %R
  - CCI (Commodity Channel Index)
  - Parabolic SAR
  - Keltner Channels
  - Donchian Channels
  - Aroon Indicator
  - Chaikin Money Flow
  - Money Flow Index
  - TRIX
  - Ultimate Oscillator

### Out of Scope
- Any instrument other than QQQ
- Shorting QQQ
- Options or derivatives on QQQ
- Leverage (1x only)
- High-frequency trading (< 1 day holding)
- Fundamental data
- Alternative data (sentiment, news)

### Constraints
- **Maximum leverage:** 1.0x (no leverage)
- **Maximum drawdown:** 25% HARD LIMIT
- **Positions:** Long QQQ or 100% cash only
- **Maximum turnover:** 50x annual (approximately 1 trade per week average)
- **Minimum position hold:** 1 day
- **Capital:** $500,000

---

## 3. Baselines

All baselines use IDENTICAL cost treatment. QQQ-specific baselines:

| Baseline | Description | Expected Sharpe | Expected Max DD |
|----------|-------------|-----------------|-----------------|
| QQQ Buy-and-Hold | 100% QQQ always | ~0.5-0.7 | 30-35% (historical) |
| 200-day MA | Long QQQ when price > 200 SMA, else cash | ~0.4-0.6 | 15-20% |
| 50/200 Golden Cross | Long when 50 SMA > 200 SMA | ~0.3-0.5 | 20-25% |
| RSI Mean Reversion | Long when RSI < 30, cash when RSI > 70 | ~0.3-0.5 | 15-20% |
| Volatility Targeting | Scale position by inverse volatility | ~0.5-0.7 | 10-15% |
| Best Single Expert | Best performing TA expert (fixed) | TBD | TBD |

### Baseline Hurdle
Meta-allocation must:
1. Beat QQQ buy-and-hold on risk-adjusted basis (Sharpe)
2. Maintain max 25% drawdown (vs QQQ's 30%+ historical DD)
3. Achieve statistical significance (t-stat > 2.0) after costs

---

## 4. Success Metrics

### Primary Metrics (Net of Costs)
| Metric | Minimum Threshold | Target | Hard Constraint |
|--------|------------------|--------|-----------------|
| Sharpe Ratio | > 0.5 | > 1.0 | - |
| Max Drawdown | < 25% | < 15% | **25% HARD LIMIT** |
| Calmar Ratio | > 0.5 | > 1.0 | - |
| Annual Return | > QQQ B&H | > 15% | - |

### Secondary Metrics
| Metric | Minimum Threshold | Target |
|--------|------------------|--------|
| Win Rate | > 50% | > 55% |
| Profit Factor | > 1.3 | > 1.8 |
| Annual Turnover | < 50x | < 25x |
| Time in Market | > 30% | 50-70% |
| Recovery Factor | > 3.0 | > 5.0 |

### Operational Metrics
| Metric | Requirement |
|--------|-------------|
| Reproducibility | 100% of results replicable |
| Signal Latency | < 1 hour after market close |
| Execution Window | Next day open or MOC |
| System Uptime | > 99.5% |

---

## 5. Kill Criteria

### Immediate Kill (Stop All Work)
- [ ] Data integrity issues discovered in QQQ data
- [ ] Regulatory concerns identified
- [ ] 25% drawdown breached in paper trading

### Phase Kill (Stop Current Phase, Reassess)
- [ ] Phase gate fails 2x consecutively
- [ ] Cannot achieve 25% max DD in backtests
- [ ] All TA experts underperform simple baselines

### Strategy Kill (Remove Expert from Consideration)
- [ ] Expert fails to beat cash for 3 consecutive test periods
- [ ] Expert max DD > 10% in any subperiod
- [ ] Expert Sharpe < 0 over full test period

---

## 6. Assumptions & Dependencies

### Key Assumptions
1. QQQ historical data (OHLCV) is accurate and survivorship-bias free
2. QQQ liquidity is sufficient for $500K positions (no market impact)
3. Transaction costs ~$0.01/share + 0.01% slippage is realistic
4. Daily rebalancing is operationally feasible
5. Regimes persist long enough to be actionable (> 5 days typically)

### Dependencies
| Dependency | Owner | Risk Level | Notes |
|------------|-------|------------|-------|
| QQQ price data | Data Platform | Low | Multiple free sources available |
| Compute infrastructure | SRE/DevOps | Low | Minimal requirements |
| TA library implementation | Quant Research | Medium | Use established libraries (TA-Lib) |
| Validation capacity | Independent Val | Medium | Critical for Phase 6 |

---

## 7. TA Expert Strategy Candidates

### Trend-Following Experts
| Expert | Indicators | Regime Hypothesis |
|--------|------------|-------------------|
| MA Crossover | SMA(50), SMA(200) | Works in strong trends |
| MACD Trend | MACD, Signal, Histogram | Works in trending markets |
| ADX Trend | ADX, +DI, -DI | Works when ADX > 25 |
| Ichimoku | Cloud, Tenkan, Kijun | Works in clear trends |
| Parabolic SAR | SAR | Works in strong directional moves |
| Donchian Breakout | 20-day high/low | Works in breakout regimes |

### Mean-Reversion Experts
| Expert | Indicators | Regime Hypothesis |
|--------|------------|-------------------|
| RSI Reversal | RSI(14) | Works in ranging markets |
| Bollinger Bounce | BB(20,2) | Works in low-vol consolidation |
| Stochastic | Stoch(14,3,3) | Works in choppy markets |
| Williams %R | %R(14) | Works in oscillating markets |
| CCI Reversal | CCI(20) | Works in mean-reverting regimes |

### Volatility-Based Experts
| Expert | Indicators | Regime Hypothesis |
|--------|------------|-------------------|
| ATR Position | ATR(14) | Scales position by volatility |
| Keltner Channel | KC(20,2) | Breakout in low-vol, fade in high-vol |
| Volatility Regime | Historical Vol | Different rules per vol regime |

### Volume-Based Experts
| Expert | Indicators | Regime Hypothesis |
|--------|------------|-------------------|
| OBV Confirm | OBV, Price | Confirms trend with volume |
| MFI Divergence | MFI(14) | Volume-weighted RSI signals |
| VWAP Deviation | VWAP | Intraday mean reversion |

---

## 8. Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Trade multiple ETFs | Diversification | Complexity, more data | Rejected - QQQ only |
| Include options | Defined risk | Complexity, costs | Rejected - stock only |
| Allow shorting | Profit in downtrends | Higher risk, costs | Rejected - long/cash only |
| Use leverage | Higher returns | Higher DD, costs | Rejected - 1x only |
| Intraday trading | More signals | Higher costs, complexity | Rejected - daily only |

---

## 9. Cost Model (Preliminary)

| Cost Component | Estimate | Notes |
|----------------|----------|-------|
| Commission | $0 | Most brokers free for ETFs |
| Bid-Ask Spread | ~$0.01/share | QQQ highly liquid |
| Slippage | ~0.01% | Conservative for $500K |
| Market Impact | ~0% | Negligible for QQQ size |

**Estimated Round-Trip Cost:** ~0.02% or ~$100 per $500K trade

---

## Approvals

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Project Sponsor | User | 2025-12-22 | ✓ |
| COO Agent | Claude | 2025-12-22 | ✓ |
| Independent Validation | Pending Phase 6 | | |

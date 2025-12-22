# Decision Log
## Adaptive Regime-Aware Trading System - QQQ Focus

### Document Control
- Last Updated: 2025-12-22

---

## Decision Format

| Field | Description |
|-------|-------------|
| ID | Unique identifier (DEC-001, DEC-002, ...) |
| Date | Decision date |
| Decision | What was decided |
| Options Considered | Alternatives that were evaluated |
| Rationale | Why this option was chosen |
| Owner | Who made/owns the decision |
| Evidence | Links to supporting analysis |
| Status | Active, Superseded, Reversed |

---

## Decisions

### DEC-001: Project Initialization
- **Date:** 2025-12-22
- **Decision:** Initialize adaptive regime-aware trading R&D program
- **Options Considered:**
  1. Full regime-aware meta-allocation (chosen)
  2. Simple ensemble without regime awareness
  3. Single best strategy approach
- **Rationale:** Hypothesis that regime-awareness can improve risk-adjusted returns is worth testing; proper phase gates and kill criteria will prevent wasted effort if hypothesis fails
- **Owner:** COO Agent
- **Evidence:** Strategy_Charter.md
- **Status:** Active

---

### DEC-002: Single Instrument Focus - QQQ
- **Date:** 2025-12-22
- **Decision:** Focus exclusively on QQQ (Invesco QQQ Trust - Nasdaq-100 ETF)
- **Options Considered:**
  1. QQQ only (chosen)
  2. Multiple ETFs (SPY, QQQ, IWM, etc.)
  3. Individual stocks
  4. Futures contracts
  5. Crypto assets
- **Rationale:**
  - QQQ is highly liquid (~$15B daily volume) - no capacity constraints for $500K
  - Single instrument simplifies research and eliminates cross-asset complexity
  - Nasdaq-100 provides tech-heavy growth exposure
  - Well-established ETF with clean historical data back to 1999
  - Lower costs than individual stocks or futures
- **Owner:** User/Sponsor
- **Evidence:** Strategy_Charter.md §2
- **Status:** Active

---

### DEC-003: Capital Allocation - $500,000
- **Date:** 2025-12-22
- **Decision:** Target capital deployment of $500,000
- **Options Considered:**
  1. $500K (chosen)
  2. $100K (smaller test)
  3. $1M+ (larger deployment)
- **Rationale:**
  - Meaningful capital to test real execution
  - Small enough relative to QQQ liquidity (~0.003% of daily volume)
  - 5% max DD = $25K max loss, acceptable risk
- **Owner:** User/Sponsor
- **Evidence:** Strategy_Charter.md §2
- **Status:** Active

---

### DEC-004: Maximum Drawdown Constraint - 25%
- **Date:** 2025-12-22 (Updated)
- **Decision:** Hard constraint of 25% maximum drawdown
- **Options Considered:**
  1. 5% max DD - $25K max loss (too restrictive)
  2. 10% max DD - $50K max loss
  3. 20% max DD - $100K max loss
  4. 25% max DD (chosen) - $125K max loss
  5. No DD constraint (accept QQQ's ~35% historical DD)
- **Rationale:**
  - 25% allows more time in market to capture QQQ upside
  - Still better than QQQ buy-and-hold (~35% max DD)
  - $125K max loss is acceptable risk for $500K portfolio
  - More realistic target that doesn't require excessive cash holdings
  - Balances capital preservation with growth potential
- **Owner:** User/Sponsor
- **Evidence:** Strategy_Charter.md §4
- **Status:** Active

---

### DEC-005: Long-Only with Cash Alternative
- **Date:** 2025-12-22
- **Decision:** Positions limited to long QQQ or 100% cash (no shorting)
- **Options Considered:**
  1. Long/cash only (chosen)
  2. Long/short QQQ
  3. Long/short with options for hedging
- **Rationale:**
  - Simpler execution and lower costs
  - No borrow costs or short squeeze risk
  - Cash provides safe haven during adverse regimes
  - Shorting increases complexity and risk without clear benefit given 5% DD constraint
- **Owner:** COO Agent
- **Evidence:** Strategy_Charter.md §2
- **Status:** Active

---

### DEC-006: No Leverage
- **Date:** 2025-12-22
- **Decision:** Maximum leverage of 1.0x (no leverage)
- **Options Considered:**
  1. 1.0x (no leverage) - chosen
  2. 1.5x leverage
  3. 2.0x leverage (TQQQ equivalent)
- **Rationale:**
  - Leverage incompatible with 25% max DD constraint
  - Leverage increases costs (margin interest)
  - Simpler risk management without leverage
  - Can always add leverage later if strategy proves robust
- **Owner:** COO Agent
- **Evidence:** Strategy_Charter.md §2
- **Status:** Active

---

### DEC-007: Daily Signal Frequency
- **Date:** 2025-12-22
- **Decision:** Generate signals daily, rebalance when signal changes
- **Options Considered:**
  1. Daily signals (chosen)
  2. Intraday signals
  3. Weekly signals
  4. Monthly signals
- **Rationale:**
  - Daily provides good balance of responsiveness vs costs
  - Most TA indicators designed for daily timeframe
  - Lower transaction costs than intraday
  - More responsive than weekly/monthly for regime changes
- **Owner:** COO Agent
- **Evidence:** Strategy_Charter.md §2
- **Status:** Active

---

### DEC-008: Comprehensive TA Expert Library
- **Date:** 2025-12-22
- **Decision:** Implement ALL major TA indicators as expert strategies
- **Options Considered:**
  1. All major TA tools (chosen) - 20+ experts
  2. Only trend-following indicators
  3. Only mean-reversion indicators
  4. Small curated set (5-10 experts)
- **Rationale:**
  - Comprehensive coverage allows testing which experts work in which regimes
  - More experts = better chance of finding regime-specific edge
  - Pre-registration prevents cherry-picking
  - Can always prune non-performing experts later
- **Owner:** User/Sponsor
- **Evidence:** Strategy_Charter.md §7
- **Status:** Active

---

## Pending Decisions

| ID | Topic | Options | Owner | Due |
|----|-------|---------|-------|-----|
| DEC-009 | QQQ data source | Yahoo Finance, Alpha Vantage, Polygon, etc. | Data Platform | Week 1 |
| DEC-010 | Backtesting framework | Custom, Backtrader, Zipline, VectorBT | Data Platform | Week 2 |
| DEC-011 | Regime detection approach | HMM, Rules-based, ML classification | ML/Stats | Week 6 |
| DEC-012 | Paper trading platform | IBKR, Alpaca, Custom simulation | Execution Eng | Week 12 |
| DEC-013 | Live broker selection | IBKR, Alpaca, Schwab, etc. | Execution Eng | Week 17 |


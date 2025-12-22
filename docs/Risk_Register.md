# Risk Register
## Adaptive Regime-Aware Trading System - QQQ Focus

### Document Control
- Last Updated: 2025-12-22

---

## Risk Rating Scale

**Impact:**
- High (H): Project failure, significant capital loss (>$125K), major delay
- Medium (M): Phase delay, partial functionality, recoverable loss (<$10K)
- Low (L): Minor delay, workaround available, minimal impact

**Likelihood:**
- High (H): > 50% probability
- Medium (M): 20-50% probability
- Low (L): < 20% probability

---

## Active Risks

### Strategic Risks

| ID | Risk Description | Impact | Likelihood | Mitigation | Owner | Status |
|----|------------------|--------|------------|------------|-------|--------|
| R001 | 25% max DD constraint challenging to achieve while significantly beating QQQ B&H | M | M | 25% is more realistic; test regime-aware allocation; monitor closely | ML/Stats | Open |
| R002 | Regime detection has insufficient accuracy for practical use | H | M | Phase 4 gate requires 60% OOS accuracy; multiple detection approaches; kill if fails | ML/Stats | Open |
| R003 | TA experts don't show regime-dependent performance | H | M | Statistical testing in Phase 4; may reject hypothesis if no regime dependence | Quant Research | Open |
| R004 | Transaction costs exceed alpha generated | M | L | QQQ very liquid; $0 commissions; stress test with 2-3x costs | Execution Eng | Open |
| R005 | Results driven by single time period (dot-com, 2008, COVID) | H | M | Subperiod stability required; walk-forward validation; test 2000-2024 | Independent Val | Open |

### Operational Risks

| ID | Risk Description | Impact | Likelihood | Mitigation | Owner | Status |
|----|------------------|--------|------------|------------|-------|--------|
| R006 | QQQ data quality issues (splits, dividends not adjusted) | H | L | Use adjusted close; verify against multiple sources; data platform owns quality | Data Platform | Open |
| R007 | Look-ahead bias in backtesting | H | M | Strict point-in-time data; independent validation verifies | Independent Val | Open |
| R008 | Overfitting to QQQ historical patterns | H | M | Walk-forward validation; out-of-sample testing; pre-registration | Quant Research | Open |
| R009 | Paper trading doesn't match live execution | M | M | Use realistic fill simulation; compare paper vs live fills | Execution Eng | Open |
| R010 | System fails to generate signal before market open | M | L | Run signal generation after market close; multiple redundancy | SRE/DevOps | Open |

### Market Risks

| ID | Risk Description | Impact | Likelihood | Mitigation | Owner | Status |
|----|------------------|--------|------------|------------|-------|--------|
| R011 | Flash crash or circuit breaker triggers 25% DD | H | L | Use stop-loss orders; fail-safe to cash; accept as black swan | Execution Eng | Open |
| R012 | QQQ regime changes faster than detection | M | M | "Tilt not switch" approach; reduce position when uncertain | ML/Stats | Open |
| R013 | Correlation breakdown between TA signals and QQQ | M | M | Monitor signal quality; fail-safe when signals unreliable | Quant Research | Open |
| R014 | QQQ delisted or structurally changed | L | L | Monitor ETF news; have contingency plan | PMO | Open |

### QQQ-Specific Risks

| ID | Risk Description | Impact | Likelihood | Mitigation | Owner | Status |
|----|------------------|--------|------------|------------|-------|--------|
| R015 | Tech sector concentration risk (QQQ = 50%+ tech) | M | H | Accept as feature not bug; regime detection should handle | Quant Research | Open |
| R016 | QQQ more volatile than SPY (higher beta) | M | H | 25% DD constraint accounts for this; size positions appropriately | ML/Stats | Open |
| R017 | QQQ dividend timing affects returns | L | M | Use total return data; adjust for dividends | Data Platform | Open |
| R018 | Large QQQ moves overnight when can't trade | M | M | Consider cash position overnight; accept residual risk | Execution Eng | Open |

### Resource Risks

| ID | Risk Description | Impact | Likelihood | Mitigation | Owner | Status |
|----|------------------|--------|------------|------------|-------|--------|
| R019 | Key personnel unavailability | M | L | Document all work; ensure knowledge transfer | PMO | Open |
| R020 | Compute resources insufficient | L | L | Cloud-based infrastructure; scale as needed | SRE/DevOps | Open |

---

## Closed/Mitigated Risks

| ID | Risk Description | Resolution | Date |
|----|------------------|------------|------|
| | | | |

---

## Risk Review Cadence
- Weekly: Review all open risks in Friday report
- Phase Gate: Comprehensive risk review before gate decision
- Incident: Add new risks immediately when identified

---

## Risk Heat Map

```
           LIKELIHOOD
           Low    Med    High
        +------+------+------+
  High  | R014 | R001 | R015 |
        |      | R002 | R016 |
IMPACT  |      | R003 |      |
        |      | R005 |      |
        +------+------+------+
  Med   | R004 | R007 |      |
        | R010 | R008 |      |
        |      | R009 |      |
        |      | R012 |      |
        |      | R013 |      |
        |      | R018 |      |
        +------+------+------+
  Low   | R019 | R017 |      |
        | R020 |      |      |
        +------+------+------+
```

**Priority Focus:** R002, R003, R005 (High Impact, Medium Likelihood)

"""
Adaptive Systematic Trading R&D - Multi-Agent System
=====================================================

This system implements a hierarchical agent architecture for managing an adaptive
systematic trading research program. The COO Agent orchestrates 7 specialized stream
leads to execute phase-gated research with strict reproducibility and cost realism.

Architecture:
- COO Orchestrator (top-level coordinator)
  ├── PMO/Governance Lead
  ├── Quant Research Lead
  ├── Data & Research Platform Lead
  ├── ML/Stats Lead (Regimes & Meta-Learning)
  ├── Execution & Trading Engineering Lead
  ├── Independent Validation Lead (VETO AUTHORITY)
  └── SRE/DevOps/Security Lead

Usage:
    python trading_rnd_agent_system.py

Requirements:
    pip install claude-agent-sdk
"""

import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition
from datetime import datetime
from pathlib import Path

# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

COO_ORCHESTRATOR_PROMPT = """
You are the Project COO AI Agent for an adaptive systematic trading R&D program.
You run the project end-to-end with phase gates, coordinate all streams/teams,
and enforce ruthless research hygiene.

## MISSION
Prove or falsify the hypothesis that different technical analysis "experts" perform
best in different market regimes, and that a regime-aware or performance-aware
meta-allocation engine can improve net-of-cost performance versus strong baselines—
first in strict walk-forward testing, then in paper trading, and only then with live capital.

## NON-NEGOTIABLE RULES (ENFORCE THESE RUTHLESSLY)
1. PHASE-GATED DISCIPLINE: Never start downstream phase unless upstream gate is PASSED with evidence links
2. REPRODUCIBILITY FIRST: Every result needs (a) dataset version/hash, (b) code hash, (c) configs/seeds
3. COST REALISM: Apply unified cost/slippage/funding/borrow model early; optimize net-of-costs
4. NO SILENT TUNING: Every parameter search logged; report trials + degrees of freedom
5. INDEPENDENT VALIDATION VETO: Validator can block paper/live trading
6. TILT NOT SWITCH: If regime confidence low, reduce turnover, revert to diversified weights
7. FAIL-SAFE OPS: Degraded data/model health → auto-fail to flat or safe baseline

## YOUR STREAM LEADS (USE Task TOOL TO DELEGATE)
- pmo-governance: Project management, phase gates, documentation
- quant-research: Expert strategies, baselines, alpha research
- data-platform: Data pipelines, research infrastructure, reproducibility
- ml-stats: Regime detection, meta-learning, calibration
- execution-engineering: Trading systems, execution, cost models
- independent-validation: Model risk, replication, VETO AUTHORITY
- sre-devops: Infrastructure, monitoring, security, incident response

## LIVING DOCUMENTS YOU MAINTAIN
Update these files continuously as the project progresses:
- Strategy_Charter.md: Hypothesis, scope, baselines, metrics, kill criteria
- Phase_Gates_Checklist.md: Deliverables + PASS/FAIL per phase with evidence
- Backlog.csv: id, stream, owner, priority, dependency, acceptance_test, due_week, status
- Decision_Log.md: date, decision, options, rationale, owner, evidence links
- Risk_Register.md: risk, impact, likelihood, mitigation, owner, status
- Experiment_Registry.md: experiment_id, hypothesis, dataset_version, code_hash, config, trials, results
- Weekly_Report.md: Progress, results, decisions, updated backlog, next week plan

## PHASES (NEVER SKIP GATES)
Phase 0: Charter + success definition
Phase 1: Literature + design-space map + replication plan
Phase 2: Data foundation + reproducible research stack
Phase 3: Expert library + baselines + unified cost model
Phase 4: Regime definitions + detectors (stability + calibration)
Phase 5: Meta-allocation engines (turnover/cost-aware)
Phase 6: Independent validation + robustness + replication proof
Phase 7: Paper trading + monitoring/runbooks + 30-day test
Phase 8: Limited live pilot (ONLY after Phase 7 PASS)

## PHASE LAUNCH PROTOCOL (MANDATORY)
For each phase N:
1. Issue prompts to relevant stream leads via Task tool
2. Create backlog tasks with owners, dependencies, acceptance tests, due weeks
3. Pre-register ALL experiments in Experiment_Registry.md BEFORE running
4. Mid-phase: Run QA/leakage/cost hygiene checks; STOP rule-violating work
5. End-phase: Gate Review with PASS/FAIL for each item + evidence links
6. If FAIL: Freeze downstream; create remediation tasks; re-run gate

## GATE REVIEW EVIDENCE BUNDLE (MINIMUM REQUIRED)
- Reproducibility proof: clean-run output, dataset hash, code hash, config/seeds
- Baselines comparison (net-of-cost) with uncertainty quantification
- Turnover + cost decomposition + sensitivity to cost shocks
- Subperiod stability (not driven by single interval)
- Failure modes + explicit falsification conditions
- Validator sign-off (when required)
- Decision log + risk register updated

## WEEKLY CADENCE (ENFORCE WITHOUT EXCEPTION)
MONDAY: Publish Weekly Plan (goals per stream, experiments queued, risks, decisions needed)
MIDWEEK: Run Hygiene Check (leakage checklist, registry completeness, cost stress tests)
FRIDAY: Publish Weekly Report (progress vs plan, results, decisions, backlog, next week)

## QUALITY CHECKLIST (RUN WEEKLY)
- Leakage: time alignment, label leakage, forward fills, survivorship
- Costs: consistent application, stress bands, funding/borrow alignment
- Overfitting: trial counts, degrees of freedom, selection bias, ablation coverage
- Operational: turnover, liquidity/capacity, venue constraints, fail-safes
- Monitoring: data health, model health, risk limits, incident readiness

## TONE
Be brutally honest. Prefer simple, defensible approaches. If edge is weak, unstable,
or dominated by costs, recommend stopping, narrowing scope, or redesigning.

## WEEK 0 DELIVERABLES (PRODUCE IMMEDIATELY)
1. Strategy_Charter.md (draft with assumptions and alternatives)
2. Phase_Gates_Checklist.md (draft)
3. Backlog.csv seeded for Weeks 1-8
4. Decision_Log.md + Risk_Register.md (initialized)
5. Weekly_Report.md (template + first weekly plan)
6. Experiment_Registry.md (initialized with first baseline experiments)
"""

PMO_GOVERNANCE_LEAD_PROMPT = """
You are the PMO/Governance Lead for an adaptive systematic trading R&D program.

## YOUR RESPONSIBILITIES
- Maintain project documentation and living files
- Track phase gates and ensure all evidence is properly linked
- Manage the backlog and ensure task dependencies are respected
- Facilitate decision-making and document rationale
- Ensure weekly cadence is followed (Monday plan, Friday report)
- Track risks and escalate blockers to COO

## DOCUMENTS YOU OWN
- Strategy_Charter.md: Keep hypothesis, scope, metrics, kill criteria current
- Phase_Gates_Checklist.md: Track all deliverables, PASS/FAIL status, evidence links
- Backlog.csv: Maintain task tracking with proper fields
- Decision_Log.md: Document all significant decisions with rationale
- Risk_Register.md: Track risks, impacts, mitigations, owners
- Weekly_Report.md: Compile weekly status from all streams

## PHASE GATE PROTOCOL
For each gate review:
1. Collect evidence from all stream leads
2. Verify reproducibility proofs are complete
3. Check that baselines comparisons include uncertainty
4. Confirm cost models were properly applied
5. Validate subperiod stability analysis exists
6. Get validator sign-off where required
7. Write PASS/FAIL determination with evidence links

## QUALITY STANDARDS
- Every backlog item needs: id, stream, owner, priority, dependency, acceptance_test, due_week, status
- Every decision needs: date, options considered, rationale, owner, evidence
- Every risk needs: description, impact (H/M/L), likelihood (H/M/L), mitigation, owner, status
- Weekly reports must compare actual vs plan, highlight blockers, preview next week

## OUTPUT FORMAT
Always provide structured, actionable outputs. Use markdown tables for tracking.
Flag any governance violations immediately to COO.
"""

QUANT_RESEARCH_LEAD_PROMPT = """
You are the Quant Research Lead for an adaptive systematic trading R&D program.

## YOUR RESPONSIBILITIES
- Design and implement technical analysis "expert" strategies
- Define and implement strong baselines (buy-and-hold, equal-weight, trend-following)
- Conduct alpha research with proper walk-forward methodology
- Document strategy logic, parameters, and expected behavior per regime
- Work with ML/Stats lead on regime-conditional performance analysis

## RESEARCH HYGIENE (NON-NEGOTIABLE)
1. PRE-REGISTER all experiments before running (Experiment_Registry.md)
2. Report ALL trials, not just successful ones
3. Use walk-forward validation only (NO in-sample optimization)
4. Apply unified cost model to ALL performance metrics
5. Test on multiple subperiods; reject strategies driven by single interval
6. Document degrees of freedom and selection bias risk

## EXPERT STRATEGY REQUIREMENTS
Each "expert" must have:
- Clear market regime hypothesis (when should it work?)
- Defined entry/exit logic with parameter bounds
- Expected turnover and capacity constraints
- Net-of-cost performance expectations
- Failure modes and conditions for deactivation

## BASELINE REQUIREMENTS
Implement these baselines with identical cost treatment:
1. Buy-and-hold benchmark (e.g., SPY for equities, BTC for crypto)
2. Equal-weight portfolio (monthly rebalance)
3. Simple trend-following (e.g., 12-1 momentum, 200-day MA)
4. Risk parity baseline

## FALSIFICATION MINDSET
Your job is to DISPROVE the hypothesis, not confirm it.
- Actively seek evidence against regime-switching value
- Test whether simple diversification beats regime-aware allocation
- Document when and why experts fail
- Recommend killing strategies that don't clear baselines net-of-costs

## DELIVERABLES
- Expert strategy specifications with code
- Baseline implementations with reproducibility proofs
- Walk-forward performance analysis (net-of-cost)
- Regime-conditional performance attribution
- Failure mode documentation
"""

DATA_PLATFORM_LEAD_PROMPT = """
You are the Data & Research Platform Lead for an adaptive systematic trading R&D program.

## YOUR RESPONSIBILITIES
- Build and maintain data pipelines with version control
- Ensure data quality, integrity, and reproducibility
- Implement the research infrastructure for experiments
- Manage dataset versioning with cryptographic hashes
- Support reproducible research with containerization

## DATA QUALITY REQUIREMENTS (NON-NEGOTIABLE)
1. Version ALL datasets with content hashes (SHA-256)
2. Document data sources, collection dates, any transformations
3. Implement survivorship bias corrections (for equities)
4. Handle corporate actions properly (splits, dividends)
5. Align timestamps correctly (no look-ahead bias)
6. Implement point-in-time correctness for all features

## REPRODUCIBILITY STACK
Every experiment must be reproducible via:
- Dataset version ID (hash)
- Code commit hash
- Configuration file with all parameters
- Random seeds for any stochastic components
- Container/environment specification

## DATA HEALTH MONITORING
Implement automated checks for:
- Missing data detection and alerting
- Outlier detection (price spikes, volume anomalies)
- Source latency monitoring
- Schema drift detection
- Backfill tracking and audit logs

## LEAKAGE PREVENTION CHECKLIST
Verify before any experiment:
[ ] No future data in features (time alignment)
[ ] No label leakage in preprocessing
[ ] Forward fills documented and appropriate
[ ] Train/validation/test splits are temporal, not random
[ ] Corporate actions don't leak future information

## INFRASTRUCTURE DELIVERABLES
- Data ingestion pipelines with monitoring
- Dataset registry with version control
- Research environment (notebooks, compute)
- Experiment tracking integration
- Artifact storage for models and results
"""

ML_STATS_LEAD_PROMPT = """
You are the ML/Stats Lead (Regimes & Meta-Learning) for an adaptive systematic trading R&D program.

## YOUR RESPONSIBILITIES
- Design and implement regime detection algorithms
- Build meta-allocation engines that are turnover/cost-aware
- Ensure statistical rigor in all modeling
- Calibrate uncertainty estimates for regime probabilities
- Implement "tilt not switch" logic for low-confidence regimes

## REGIME DETECTION REQUIREMENTS
1. Define regimes with clear, interpretable criteria
2. Regimes must be detectable in real-time (no look-ahead)
3. Measure regime persistence and transition probabilities
4. Calibrate regime probability estimates (reliability diagrams)
5. Test regime stability across time periods
6. Document regime detector failure modes

## META-ALLOCATION ENGINE REQUIREMENTS
1. Cost-aware: penalize turnover in objective function
2. Uncertainty-aware: reduce position sizes when confidence is low
3. "Tilt not switch": gradual transitions, not binary regime switches
4. Robust to regime misclassification
5. Graceful degradation when regime detector is uncertain

## STATISTICAL RIGOR (NON-NEGOTIABLE)
- Report confidence intervals, not just point estimates
- Use proper multiple testing corrections
- Document all modeling assumptions
- Test for non-stationarity and structural breaks
- Validate out-of-sample, never in-sample
- Report degrees of freedom in all optimizations

## CALIBRATION REQUIREMENTS
Regime probabilities must be well-calibrated:
- Reliability diagrams showing calibration quality
- Brier scores for probabilistic predictions
- Comparison to climatological baselines
- Calibration stability across time periods

## OVERFITTING PREVENTION
- Pre-register all experiments
- Report total trials, not just successes
- Use information criteria (AIC, BIC) for model selection
- Prefer simpler models unless complexity is justified OOS
- Ablation studies for all model components

## DELIVERABLES
- Regime detection algorithms with calibration proof
- Meta-allocation engine with cost-awareness
- Statistical validation reports
- Uncertainty quantification methodology
- Ablation study results
"""

EXECUTION_ENGINEERING_LEAD_PROMPT = """
You are the Execution & Trading Engineering Lead for an adaptive systematic trading R&D program.

## YOUR RESPONSIBILITIES
- Build the unified cost/slippage/funding/borrow model
- Implement execution algorithms and venue integration
- Design the trading system architecture
- Ensure operational robustness and fail-safes
- Support paper trading and live trading infrastructure

## UNIFIED COST MODEL (CRITICAL)
Implement a comprehensive cost model including:
1. Commission/fees by venue
2. Bid-ask spread (use realistic estimates, not midpoint)
3. Market impact (function of size vs ADV)
4. Slippage model (execution uncertainty)
5. Funding costs (for leveraged positions)
6. Borrow costs (for short positions)
7. Latency costs (for time-sensitive strategies)

## COST MODEL REQUIREMENTS
- Apply SAME cost model across ALL strategies and baselines
- Implement cost stress testing (2x, 3x cost scenarios)
- Track cost attribution by component
- Weekly cost sensitivity analysis
- Document cost assumptions and update quarterly

## TRADING SYSTEM ARCHITECTURE
- Order management system (OMS)
- Execution management system (EMS)
- Position and P&L tracking
- Risk limit enforcement
- Venue connectivity (paper and live)

## FAIL-SAFE REQUIREMENTS (NON-NEGOTIABLE)
Implement automatic fail-safe to flat (or safe baseline) when:
- Data health is degraded (stale prices, missing data)
- Model health is degraded (anomalous signals)
- Risk limits are breached
- System connectivity is impaired
- Manual override is triggered

## OPERATIONAL REQUIREMENTS
- Turnover monitoring and alerts
- Capacity/liquidity checks before trading
- Pre-trade risk checks
- Post-trade reconciliation
- Incident logging and alerting

## PAPER TRADING REQUIREMENTS
- Realistic fill simulation (not optimistic)
- Latency simulation
- Cost model application identical to live
- Full audit trail
- Comparison to live market execution (when available)

## DELIVERABLES
- Unified cost model implementation
- Trading system architecture
- Fail-safe implementation
- Paper trading infrastructure
- Execution monitoring dashboards
"""

INDEPENDENT_VALIDATION_LEAD_PROMPT = """
You are the Independent Validation / Model Risk Lead for an adaptive systematic trading R&D program.

## YOUR AUTHORITY
YOU HAVE VETO POWER over paper and live trading transitions. Your sign-off is REQUIRED
before any capital deployment. This authority cannot be overridden by other leads.

## YOUR RESPONSIBILITIES
- Independent replication of all research results
- Model risk assessment and documentation
- Challenge assumptions and methodology
- Verify reproducibility from clean checkout
- Sign-off on phase gates (especially Phases 6, 7, 8)

## VALIDATION METHODOLOGY (NON-NEGOTIABLE)
For each strategy/model requiring sign-off:
1. REPLICATE results from clean environment using only documented artifacts
2. VERIFY dataset version matches claimed hash
3. VERIFY code hash matches claimed version
4. RUN with documented config/seeds, compare outputs
5. TEST on held-out periods not used in development
6. STRESS TEST with varied cost assumptions
7. CHALLENGE key assumptions with adversarial scenarios

## WHAT TO LOOK FOR (RED FLAGS)
- Results that don't replicate within tolerance
- Missing or incomplete experiment registry entries
- Suspiciously good Sharpe ratios (>2 sustained)
- Performance driven by single time period
- Excessive parameter tuning without documentation
- Cost assumptions that seem optimistic
- Regime detector that's too accurate (possible look-ahead)
- Insufficient degrees of freedom reporting

## REPLICATION PROOF REQUIREMENTS
Your sign-off requires documented evidence of:
- Clean-environment reproduction (new container/VM)
- Exact match of outputs (within numerical tolerance)
- Dataset hash verification
- Code hash verification
- Config/seed verification
- Subperiod stability analysis
- Cost sensitivity analysis

## VETO CONDITIONS
You MUST veto progression if:
- Results don't replicate
- Reproducibility artifacts are incomplete
- Evidence of data leakage
- Unrealistic cost assumptions
- Insufficient out-of-sample testing
- Regime detector not properly calibrated
- Failure modes not documented
- Risk register not updated

## DELIVERABLES
- Replication reports with evidence
- Model risk assessments
- Validation sign-off documents (or veto memos)
- Challenge documentation
- Independent stress test results
"""

SRE_DEVOPS_LEAD_PROMPT = """
You are the SRE/DevOps/Security Lead for an adaptive systematic trading R&D program.

## YOUR RESPONSIBILITIES
- Infrastructure provisioning and management
- Monitoring, alerting, and incident response
- Security and access control
- CI/CD pipelines for research and trading code
- Disaster recovery and business continuity

## INFRASTRUCTURE REQUIREMENTS
- Reproducible environments (containers, IaC)
- Compute resources for research (GPU if needed)
- Storage for datasets and artifacts
- Network security and isolation
- Backup and recovery systems

## MONITORING REQUIREMENTS
Implement monitoring for:
1. Data health (freshness, completeness, quality)
2. Model health (signal distributions, prediction quality)
3. System health (latency, errors, resource usage)
4. Trading health (fills, slippage, P&L)
5. Risk metrics (exposure, drawdown, VAR)

## ALERTING TIERS
- P1 (Critical): Trading halt required, immediate response
- P2 (High): Degraded operation, response within 1 hour
- P3 (Medium): Non-critical issue, response within 4 hours
- P4 (Low): Informational, address during business hours

## INCIDENT RESPONSE
- Runbooks for common failure scenarios
- Escalation paths defined
- Post-incident review process
- Incident tracking and metrics

## SECURITY REQUIREMENTS
- Secrets management (API keys, credentials)
- Access control (least privilege)
- Audit logging for all sensitive operations
- Network segmentation (research vs trading)
- Vulnerability scanning and patching

## CI/CD REQUIREMENTS
- Automated testing for all code changes
- Code review requirements
- Deployment pipelines (dev → staging → prod)
- Rollback procedures
- Version control for all artifacts

## FAIL-SAFE INTEGRATION
Work with Execution Engineering to ensure:
- Health checks trigger fail-safe when appropriate
- Monitoring covers all fail-safe conditions
- Alerts fire before fail-safe triggers (when possible)
- Fail-safe activations are logged and reviewed

## DELIVERABLES
- Infrastructure-as-code specifications
- Monitoring and alerting configuration
- Runbooks for incident response
- Security documentation
- CI/CD pipeline specifications
"""

# =============================================================================
# AGENT DEFINITIONS FOR SDK
# =============================================================================

def get_agent_definitions() -> dict[str, AgentDefinition]:
    """Return all agent definitions for the trading R&D system."""
    return {
        "pmo-governance": AgentDefinition(
            description="PMO/Governance Lead - Project documentation, phase gates, backlog management, decision tracking, risk register, weekly reporting. Use for governance, documentation, and process enforcement.",
            prompt=PMO_GOVERNANCE_LEAD_PROMPT,
            tools=["Read", "Write", "Edit", "Glob", "Grep"],
            model="sonnet"
        ),
        "quant-research": AgentDefinition(
            description="Quant Research Lead - Expert strategy design, baselines, alpha research, walk-forward testing. Use for strategy development and performance analysis.",
            prompt=QUANT_RESEARCH_LEAD_PROMPT,
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            model="sonnet"
        ),
        "data-platform": AgentDefinition(
            description="Data & Research Platform Lead - Data pipelines, versioning, reproducibility infrastructure, data quality. Use for data engineering and research infrastructure.",
            prompt=DATA_PLATFORM_LEAD_PROMPT,
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            model="sonnet"
        ),
        "ml-stats": AgentDefinition(
            description="ML/Stats Lead - Regime detection, meta-learning, calibration, statistical validation. Use for regime models and meta-allocation engines.",
            prompt=ML_STATS_LEAD_PROMPT,
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            model="sonnet"
        ),
        "execution-engineering": AgentDefinition(
            description="Execution & Trading Engineering Lead - Cost models, trading systems, execution, fail-safes. Use for trading infrastructure and cost modeling.",
            prompt=EXECUTION_ENGINEERING_LEAD_PROMPT,
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            model="sonnet"
        ),
        "independent-validation": AgentDefinition(
            description="Independent Validation Lead - VETO AUTHORITY. Replication, model risk, sign-off on phase gates. Use for validation and must sign-off before paper/live trading.",
            prompt=INDEPENDENT_VALIDATION_LEAD_PROMPT,
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            model="opus"  # Most capable model for critical validation
        ),
        "sre-devops": AgentDefinition(
            description="SRE/DevOps/Security Lead - Infrastructure, monitoring, alerting, security, CI/CD. Use for operational infrastructure and incident response.",
            prompt=SRE_DEVOPS_LEAD_PROMPT,
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            model="sonnet"
        )
    }


# =============================================================================
# LIVING DOCUMENT TEMPLATES
# =============================================================================

STRATEGY_CHARTER_TEMPLATE = """# Strategy Charter
## Adaptive Regime-Aware Trading System

### Document Control
- Version: 0.1 (Draft)
- Last Updated: {date}
- Status: DRAFT - Pending Phase 0 Gate

---

## 1. Hypothesis

**Primary Hypothesis:**
Different technical analysis "expert" strategies perform best in different market regimes,
and a regime-aware or performance-aware meta-allocation engine can improve net-of-cost
performance versus strong baselines.

**Sub-Hypotheses:**
1. Market regimes are detectable in real-time with sufficient accuracy
2. Expert performance is regime-dependent (not just random variation)
3. Meta-allocation can capture regime-expert relationships without excessive turnover
4. Net-of-cost performance exceeds diversified baselines

### Falsification Conditions
The hypothesis is REJECTED if:
- [ ] Regime detection accuracy < 60% (out-of-sample)
- [ ] Expert performance not statistically different across regimes (p > 0.05)
- [ ] Meta-allocation Sharpe < Best baseline Sharpe (net-of-cost) by > 0.1
- [ ] Turnover costs exceed alpha generated
- [ ] Results driven by single time period (fails subperiod stability)

---

## 2. Scope

### In Scope
- Asset class: [TBD - e.g., US Equities, Crypto, Futures]
- Universe: [TBD - e.g., S&P 500, Top 100 crypto by market cap]
- Time horizon: [TBD - e.g., Daily rebalancing, intraday signals]
- Regime types: [TBD - e.g., Trend, Mean-reversion, High-vol, Low-vol]
- Expert count: [TBD - e.g., 5-10 distinct strategies]

### Out of Scope
- High-frequency trading (< 1 minute holding periods)
- Exotic derivatives
- Illiquid assets (< $1M daily volume)
- Fundamental/alternative data (Phase 1 only uses price/volume)

### Constraints
- Maximum leverage: [TBD]
- Maximum turnover: [TBD - e.g., 200% annual]
- Minimum capacity: [TBD - e.g., $10M]
- Maximum drawdown tolerance: [TBD - e.g., 20%]

---

## 3. Baselines

All baselines use IDENTICAL cost treatment as test strategies.

| Baseline | Description | Expected Sharpe (net) |
|----------|-------------|----------------------|
| Buy-and-Hold | Benchmark index (e.g., SPY) | ~0.4 |
| Equal-Weight | Monthly rebalanced equal-weight | ~0.5 |
| Trend-Following | 12-1 momentum or 200-day MA | ~0.3-0.5 |
| Risk Parity | Inverse-vol weighted | ~0.5-0.6 |
| Best Single Expert | Best performing expert (fixed) | TBD |

### Baseline Hurdle
Meta-allocation must beat BEST baseline by statistically significant margin
(t-stat > 2.0) after costs to be considered successful.

---

## 4. Success Metrics

### Primary Metrics (Net of Costs)
| Metric | Minimum Threshold | Target |
|--------|------------------|--------|
| Sharpe Ratio | > Best Baseline + 0.1 | > 0.7 |
| Information Ratio | > 0.3 | > 0.5 |
| Max Drawdown | < 25% | < 15% |
| Calmar Ratio | > 0.5 | > 1.0 |

### Secondary Metrics
| Metric | Minimum Threshold | Target |
|--------|------------------|--------|
| Win Rate | > 50% | > 55% |
| Profit Factor | > 1.2 | > 1.5 |
| Annual Turnover | < 400% | < 200% |
| Hit Rate by Regime | > 55% per regime | > 60% |

### Operational Metrics
| Metric | Requirement |
|--------|-------------|
| Reproducibility | 100% of results replicable |
| Data Latency | < 15 minutes |
| Execution Latency | < 1 second |
| System Uptime | > 99.5% |

---

## 5. Kill Criteria

### Immediate Kill (Stop All Work)
- [ ] Fraud or data integrity issues discovered
- [ ] Regulatory concerns identified
- [ ] Key personnel unavailable for > 2 weeks

### Phase Kill (Stop Current Phase, Reassess)
- [ ] Phase gate fails 2x consecutively
- [ ] Core assumption invalidated
- [ ] Cost model reveals negative expected returns

### Strategy Kill (Remove from Consideration)
- [ ] Strategy fails to beat baseline for 3 consecutive test periods
- [ ] Strategy capacity < minimum threshold
- [ ] Strategy drawdown exceeds 2x expectation

---

## 6. Assumptions & Dependencies

### Key Assumptions
1. Historical data quality is sufficient for strategy development
2. Market microstructure is stable enough for backtesting validity
3. Transaction costs are estimable with reasonable accuracy
4. Regimes are persistent enough to be actionable

### Dependencies
| Dependency | Owner | Risk Level |
|------------|-------|------------|
| Market data access | Data Platform | Medium |
| Compute infrastructure | SRE/DevOps | Low |
| Domain expertise | Quant Research | Medium |
| Validation capacity | Independent Val | High |

---

## 7. Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Single best expert | Simple, low turnover | No adaptation | Baseline only |
| Equal-weight experts | Diversified | No regime awareness | Baseline only |
| Pure ML regime detection | Flexible | Overfit risk, black box | Test in Phase 4 |
| Rules-based regime detection | Interpretable | May miss regimes | Test in Phase 4 |

---

## Approvals

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Project Sponsor | | | |
| COO Agent | | | |
| Independent Validation | | | |
"""

PHASE_GATES_CHECKLIST_TEMPLATE = """# Phase Gates Checklist
## Adaptive Regime-Aware Trading System

### Document Control
- Version: 0.1 (Draft)
- Last Updated: {date}
- Status: DRAFT

---

## Phase 0: Charter + Success Definition

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Strategy_Charter.md complete | PMO | ⬜ PENDING | |
| Hypothesis clearly stated | Quant Research | ⬜ PENDING | |
| Baselines defined | Quant Research | ⬜ PENDING | |
| Success metrics defined | PMO | ⬜ PENDING | |
| Kill criteria defined | PMO | ⬜ PENDING | |
| Scope boundaries clear | COO | ⬜ PENDING | |

### Gate Criteria
- [ ] All deliverables complete
- [ ] Stakeholder sign-off obtained
- [ ] Risk register initialized

### Gate Status: ⬜ NOT STARTED

---

## Phase 1: Literature + Design-Space Map + Replication Plan

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Literature review complete | Quant Research | ⬜ PENDING | |
| Regime detection survey | ML/Stats | ⬜ PENDING | |
| Expert strategy candidates listed | Quant Research | ⬜ PENDING | |
| Design space documented | Quant Research | ⬜ PENDING | |
| Replication plan for key papers | Independent Val | ⬜ PENDING | |
| Initial risk assessment | PMO | ⬜ PENDING | |

### Gate Criteria
- [ ] At least 20 relevant papers reviewed
- [ ] 10+ expert strategy candidates identified
- [ ] 3+ regime detection approaches identified
- [ ] Replication targets selected (2-3 papers)

### Gate Status: ⬜ NOT STARTED

---

## Phase 2: Data Foundation + Reproducible Research Stack

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Data sources identified | Data Platform | ⬜ PENDING | |
| Data pipelines implemented | Data Platform | ⬜ PENDING | |
| Dataset versioning system | Data Platform | ⬜ PENDING | |
| Research environment setup | Data Platform | ⬜ PENDING | |
| Reproducibility framework | Data Platform | ⬜ PENDING | |
| Data quality checks | Data Platform | ⬜ PENDING | |
| Leakage prevention verified | Independent Val | ⬜ PENDING | |

### Gate Criteria
- [ ] Data pipelines operational with monitoring
- [ ] All datasets versioned with hashes
- [ ] Research environment reproducible from scratch
- [ ] Leakage checklist passed
- [ ] Data quality baseline established

### Gate Status: ⬜ NOT STARTED

---

## Phase 3: Expert Library + Baselines + Unified Cost Model

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Unified cost model implemented | Execution Eng | ⬜ PENDING | |
| Cost model validated | Independent Val | ⬜ PENDING | |
| All baselines implemented | Quant Research | ⬜ PENDING | |
| Baseline performance documented | Quant Research | ⬜ PENDING | |
| Expert strategies implemented | Quant Research | ⬜ PENDING | |
| Expert performance documented | Quant Research | ⬜ PENDING | |
| All experiments pre-registered | PMO | ⬜ PENDING | |

### Gate Criteria
- [ ] Cost model covers all cost components
- [ ] Cost stress tests completed (2x, 3x scenarios)
- [ ] All baselines have reproducible results
- [ ] At least 5 expert strategies implemented
- [ ] Walk-forward validation used throughout
- [ ] All experiments in registry with results

### Gate Status: ⬜ NOT STARTED

---

## Phase 4: Regime Definitions + Detectors (Stability + Calibration)

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Regime definitions documented | ML/Stats | ⬜ PENDING | |
| Regime detector implemented | ML/Stats | ⬜ PENDING | |
| Regime detector calibrated | ML/Stats | ⬜ PENDING | |
| Stability analysis complete | ML/Stats | ⬜ PENDING | |
| Expert-regime mapping analyzed | Quant Research | ⬜ PENDING | |
| No look-ahead verified | Independent Val | ⬜ PENDING | |

### Gate Criteria
- [ ] Regime definitions are interpretable
- [ ] Regime detector works in real-time (no look-ahead)
- [ ] Calibration diagrams show good calibration
- [ ] Regime persistence > random baseline
- [ ] Expert performance differs by regime (stat sig)

### Gate Status: ⬜ NOT STARTED

---

## Phase 5: Meta-Allocation Engines (Turnover/Cost-Aware)

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Meta-allocation engine implemented | ML/Stats | ⬜ PENDING | |
| Cost-awareness verified | Execution Eng | ⬜ PENDING | |
| Turnover constraints implemented | ML/Stats | ⬜ PENDING | |
| "Tilt not switch" logic verified | ML/Stats | ⬜ PENDING | |
| Performance vs baselines (net) | Quant Research | ⬜ PENDING | |
| Ablation studies complete | ML/Stats | ⬜ PENDING | |

### Gate Criteria
- [ ] Meta-allocation respects turnover constraints
- [ ] Cost-aware objective function verified
- [ ] "Tilt not switch" behavior demonstrated
- [ ] Outperforms baselines net-of-cost
- [ ] Ablation shows each component adds value

### Gate Status: ⬜ NOT STARTED

---

## Phase 6: Independent Validation + Robustness + Replication Proof

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Full replication from clean env | Independent Val | ⬜ PENDING | |
| Robustness tests complete | Independent Val | ⬜ PENDING | |
| Cost sensitivity analysis | Independent Val | ⬜ PENDING | |
| Subperiod stability verified | Independent Val | ⬜ PENDING | |
| Failure modes documented | Independent Val | ⬜ PENDING | |
| Model risk assessment | Independent Val | ⬜ PENDING | |

### Gate Criteria
- [ ] All results replicate within tolerance
- [ ] All hashes verified (data, code, config)
- [ ] Cost sensitivity acceptable (survives 2x costs)
- [ ] No single period drives results
- [ ] Failure modes understood and mitigated
- [ ] VALIDATOR SIGN-OFF OBTAINED

### Gate Status: ⬜ NOT STARTED

---

## Phase 7: Paper Trading Integration + Monitoring/Runbooks + 30-Day Test

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Paper trading system operational | Execution Eng | ⬜ PENDING | |
| Monitoring dashboards live | SRE/DevOps | ⬜ PENDING | |
| Alerting configured | SRE/DevOps | ⬜ PENDING | |
| Runbooks complete | SRE/DevOps | ⬜ PENDING | |
| Fail-safe tested | Execution Eng | ⬜ PENDING | |
| 30-day paper trading complete | Execution Eng | ⬜ PENDING | |
| Paper vs backtest comparison | Independent Val | ⬜ PENDING | |

### Gate Criteria
- [ ] Paper trading matches backtest within tolerance
- [ ] All fail-safes tested and working
- [ ] Monitoring covers all health dimensions
- [ ] No critical incidents during 30-day test
- [ ] VALIDATOR SIGN-OFF OBTAINED

### Gate Status: ⬜ NOT STARTED

---

## Phase 8: Limited Live Pilot + Scale Governance (CONDITIONAL)

### Prerequisites
- Phase 7 PASSED with validator sign-off
- Capital allocation approved
- Legal/compliance review complete

### Deliverables
| Item | Owner | Status | Evidence Link |
|------|-------|--------|---------------|
| Live trading system operational | Execution Eng | ⬜ PENDING | |
| Risk limits implemented | Execution Eng | ⬜ PENDING | |
| Incident response tested | SRE/DevOps | ⬜ PENDING | |
| Scale governance defined | PMO | ⬜ PENDING | |
| Live performance tracking | Quant Research | ⬜ PENDING | |

### Gate Criteria
- [ ] Live performance within expected range
- [ ] No risk limit breaches
- [ ] Operational stability demonstrated
- [ ] Scale criteria defined for expansion

### Gate Status: ⬜ NOT STARTED (Requires Phase 7 PASS)

---

## Gate Status Legend
- ⬜ NOT STARTED
- 🔄 IN PROGRESS
- ✅ PASSED
- ❌ FAILED
- 🔒 BLOCKED
"""

BACKLOG_TEMPLATE = """id,stream,owner,priority,dependency,acceptance_test,due_week,status
1,PMO,pmo-governance,P0,,Strategy_Charter.md complete and reviewed,1,pending
2,PMO,pmo-governance,P0,,Phase_Gates_Checklist.md complete,1,pending
3,PMO,pmo-governance,P0,,Backlog.csv seeded for weeks 1-8,1,pending
4,PMO,pmo-governance,P0,,Decision_Log.md initialized,1,pending
5,PMO,pmo-governance,P0,,Risk_Register.md initialized,1,pending
6,PMO,pmo-governance,P0,,Weekly_Report.md template created,1,pending
7,PMO,pmo-governance,P0,,Experiment_Registry.md initialized,1,pending
8,Quant,quant-research,P1,1,Literature review - 20+ papers reviewed,2,pending
9,Quant,quant-research,P1,8,Expert strategy candidates - 10+ identified,2,pending
10,ML,ml-stats,P1,8,Regime detection approaches - 3+ identified,2,pending
11,Data,data-platform,P1,,Data sources identified and documented,2,pending
12,Data,data-platform,P1,11,Data pipeline architecture designed,3,pending
13,Data,data-platform,P2,12,Dataset versioning system implemented,3,pending
14,Data,data-platform,P2,13,Research environment reproducible,4,pending
15,Quant,quant-research,P1,14,Baselines implemented with cost model,4,pending
16,Exec,execution-engineering,P1,11,Unified cost model designed,3,pending
17,Exec,execution-engineering,P1,16,Cost model implemented,4,pending
18,Validation,independent-validation,P1,17,Cost model validated,4,pending
19,Quant,quant-research,P2,15;17,Expert strategies implemented (5+),5,pending
20,ML,ml-stats,P2,14;19,Regime detector v1 implemented,5,pending
21,ML,ml-stats,P2,20,Regime detector calibrated,6,pending
22,Validation,independent-validation,P1,21,Regime detector no look-ahead verified,6,pending
23,ML,ml-stats,P2,21;22,Meta-allocation engine v1,6,pending
24,Quant,quant-research,P2,23,Performance analysis vs baselines,7,pending
25,Validation,independent-validation,P0,24,Full replication from clean environment,7,pending
26,Validation,independent-validation,P0,25,Robustness tests complete,8,pending
27,Exec,execution-engineering,P2,26,Paper trading system operational,8,pending
28,SRE,sre-devops,P2,27,Monitoring and alerting configured,8,pending
"""

DECISION_LOG_TEMPLATE = """# Decision Log
## Adaptive Regime-Aware Trading System

### Document Control
- Last Updated: {date}

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
- **Date:** {date}
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

## Pending Decisions

| ID | Topic | Options | Owner | Due |
|----|-------|---------|-------|-----|
| | | | | |

"""

RISK_REGISTER_TEMPLATE = """# Risk Register
## Adaptive Regime-Aware Trading System

### Document Control
- Last Updated: {date}

---

## Risk Rating Scale

**Impact:**
- High (H): Project failure, significant capital loss, major delay
- Medium (M): Phase delay, partial functionality, recoverable loss
- Low (L): Minor delay, workaround available, minimal impact

**Likelihood:**
- High (H): > 50% probability
- Medium (M): 20-50% probability
- Low (L): < 20% probability

---

## Active Risks

| ID | Risk Description | Impact | Likelihood | Mitigation | Owner | Status |
|----|------------------|--------|------------|------------|-------|--------|
| R001 | Regime detection has insufficient accuracy for practical use | H | M | Phase 4 gate requires minimum accuracy threshold; kill criteria defined | ML/Stats | Open |
| R002 | Transaction costs exceed alpha generated | H | M | Unified cost model applied from Phase 3; cost stress testing required | Execution Eng | Open |
| R003 | Results driven by single time period (overfitting) | H | M | Subperiod stability required at all gates; walk-forward only | Independent Val | Open |
| R004 | Data quality issues (survivorship, look-ahead) | H | L | Leakage checklist enforced; data platform owns quality | Data Platform | Open |
| R005 | Key personnel unavailability | M | L | Document all work; ensure knowledge transfer | PMO | Open |
| R006 | Infrastructure reliability during paper/live trading | M | M | Fail-safe to flat implemented; monitoring required | SRE/DevOps | Open |
| R007 | Regulatory or compliance issues discovered | H | L | Early legal review; conservative position limits | PMO | Open |

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

"""

EXPERIMENT_REGISTRY_TEMPLATE = """# Experiment Registry
## Adaptive Regime-Aware Trading System

### Document Control
- Last Updated: {date}

---

## Registry Rules (NON-NEGOTIABLE)

1. **PRE-REGISTER** all experiments BEFORE running
2. Record ALL trials, not just successful ones
3. Include exact dataset version (hash)
4. Include exact code version (commit hash)
5. Include full configuration with seeds
6. Link to results (even if negative)

---

## Experiment Format

| Field | Description |
|-------|-------------|
| ID | EXP-XXX format |
| Date | Registration date |
| Hypothesis | What are we testing? |
| Dataset Version | SHA-256 hash of dataset |
| Code Hash | Git commit hash |
| Config | Full configuration including seeds |
| Trials | Number of trials/variations |
| Results Link | Path to results file |
| Status | Registered, Running, Complete, Failed |
| Conclusion | What did we learn? |

---

## Registered Experiments

### EXP-001: Baseline Performance Establishment
- **Date:** {date}
- **Hypothesis:** Establish baseline performance metrics for comparison
- **Dataset Version:** [PENDING - to be filled when data ready]
- **Code Hash:** [PENDING - to be filled when implemented]
- **Config:**
  ```yaml
  baselines:
    - buy_and_hold
    - equal_weight_monthly
    - trend_following_12_1
    - risk_parity
  period: [PENDING]
  cost_model: unified_v1
  seeds: [42, 123, 456]
  ```
- **Trials:** 4 baselines × 3 seeds = 12
- **Results Link:** [PENDING]
- **Status:** Registered
- **Conclusion:** [PENDING]

---

## Experiment Queue

| ID | Hypothesis | Owner | Priority | Dependencies |
|----|------------|-------|----------|--------------|
| EXP-002 | Expert strategy performance by regime | Quant Research | P1 | EXP-001 |
| EXP-003 | Regime detection accuracy | ML/Stats | P1 | Data ready |
| EXP-004 | Meta-allocation vs baselines | ML/Stats | P2 | EXP-001, EXP-003 |

---

## Completed Experiments Summary

| ID | Hypothesis | Result | Key Finding |
|----|------------|--------|-------------|
| | | | |

"""

WEEKLY_REPORT_TEMPLATE = """# Weekly Report
## Adaptive Regime-Aware Trading System

### Report Period: Week {week_number}
### Date: {date}

---

## Executive Summary

[2-3 sentence summary of week's progress, key decisions, and blockers]

---

## Progress vs Plan

### Planned This Week
| Task | Owner | Status | Notes |
|------|-------|--------|-------|
| | | | |

### Completed This Week
| Task | Owner | Evidence |
|------|-------|----------|
| | | |

### Carried Forward
| Task | Owner | Reason | New Due |
|------|-------|--------|---------|
| | | | |

---

## Key Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Tasks completed | | | |
| Experiments registered | | | |
| Experiments completed | | | |
| Blockers resolved | | | |

---

## Decisions Made

| Decision | Rationale | Owner |
|----------|-----------|-------|
| | | |

(Full details in Decision_Log.md)

---

## Risks & Issues

### New Risks Identified
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| | | | |

### Active Blockers
| Blocker | Impact | Owner | Resolution Plan |
|---------|--------|-------|-----------------|
| | | | |

(Full details in Risk_Register.md)

---

## Phase Gate Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Charter | ⬜ NOT STARTED | |
| Phase 1: Literature | ⬜ NOT STARTED | |
| Phase 2: Data Foundation | ⬜ NOT STARTED | |
| Phase 3: Expert Library | ⬜ NOT STARTED | |
| Phase 4: Regime Detection | ⬜ NOT STARTED | |
| Phase 5: Meta-Allocation | ⬜ NOT STARTED | |
| Phase 6: Validation | ⬜ NOT STARTED | |
| Phase 7: Paper Trading | ⬜ NOT STARTED | |

---

## Next Week Plan

### Goals
1.
2.
3.

### Experiments Queued
| Experiment ID | Hypothesis | Owner |
|---------------|------------|-------|
| | | |

### Decisions Needed
| Topic | Options | Owner | Deadline |
|-------|---------|-------|----------|
| | | | |

---

## Hygiene Check Results

### Leakage Checklist
- [ ] Time alignment verified
- [ ] No label leakage
- [ ] Forward fills documented
- [ ] Survivorship handled (if equities)

### Experiment Registry
- [ ] All experiments pre-registered
- [ ] All completed experiments have results
- [ ] No "silent tuning" detected

### Cost Model
- [ ] Costs applied consistently
- [ ] Stress tests current (2x, 3x scenarios)

---

## Appendix

### Stream Status
| Stream | Lead | Status | Key Items |
|--------|------|--------|-----------|
| PMO/Governance | pmo-governance | | |
| Quant Research | quant-research | | |
| Data Platform | data-platform | | |
| ML/Stats | ml-stats | | |
| Execution Eng | execution-engineering | | |
| Validation | independent-validation | | |
| SRE/DevOps | sre-devops | | |

"""


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def initialize_project():
    """Initialize the project with Week 0 deliverables."""
    from datetime import datetime

    date_str = datetime.now().strftime("%Y-%m-%d")

    # Create docs directory
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    # Write all templates
    templates = [
        ("Strategy_Charter.md", STRATEGY_CHARTER_TEMPLATE),
        ("Phase_Gates_Checklist.md", PHASE_GATES_CHECKLIST_TEMPLATE),
        ("Backlog.csv", BACKLOG_TEMPLATE),
        ("Decision_Log.md", DECISION_LOG_TEMPLATE),
        ("Risk_Register.md", RISK_REGISTER_TEMPLATE),
        ("Experiment_Registry.md", EXPERIMENT_REGISTRY_TEMPLATE),
        ("Weekly_Report.md", WEEKLY_REPORT_TEMPLATE),
    ]

    for filename, template in templates:
        filepath = docs_dir / filename
        content = template.format(date=date_str, week_number=0)
        filepath.write_text(content)
        print(f"Created: {filepath}")

    print("\nWeek 0 deliverables initialized!")
    print("Run the agent system to begin Phase 0.")


async def run_coo_agent(prompt: str):
    """Run the COO orchestrator agent with the full team."""

    print(f"\n{'='*60}")
    print("ADAPTIVE TRADING R&D - COO AGENT SYSTEM")
    print(f"{'='*60}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'='*60}\n")

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=COO_ORCHESTRATOR_PROMPT,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task"],
            agents=get_agent_definitions(),
            permission_mode="acceptEdits"
        )
    ):
        # Log agent activity
        if hasattr(message, 'content'):
            for block in message.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    if block.name == 'Task':
                        agent_type = block.input.get('subagent_type', 'unknown')
                        print(f"[COO] Delegating to: {agent_type}")
                elif hasattr(block, 'text'):
                    print(block.text)

        # Final result
        if hasattr(message, "result"):
            print(f"\n{'='*60}")
            print("SESSION COMPLETE")
            print(f"{'='*60}")
            print(message.result)


async def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "init":
            await initialize_project()
        else:
            # Run with custom prompt
            prompt = " ".join(sys.argv[1:])
            await run_coo_agent(prompt)
    else:
        # Default: Initialize and start Phase 0
        await initialize_project()
        await run_coo_agent(
            "Begin Phase 0. Create all Week 0 deliverables: "
            "Strategy Charter, Phase Gates Checklist, Backlog, "
            "Decision Log, Risk Register, Experiment Registry, "
            "and Weekly Report. Delegate to appropriate stream leads."
        )


if __name__ == "__main__":
    asyncio.run(main())

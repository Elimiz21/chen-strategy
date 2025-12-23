"""
Regime Detection Module
=======================

Implements 4 regime detection approaches from Phase 1 survey:
1. Rules-Based (Moving Average)
2. Hidden Markov Model (HMM)
3. Threshold Model (Volatility-Based)
4. Hybrid (Rules + ML Confirmation)
"""

from .detector import (
    RegimeDetector,
    RulesBasedDetector,
    ThresholdDetector,
    HMMDetector,
    HybridDetector,
    Regime,
)

__all__ = [
    "RegimeDetector",
    "RulesBasedDetector",
    "ThresholdDetector",
    "HMMDetector",
    "HybridDetector",
    "Regime",
]

"""
Expert Trading Strategies Module
================================

Contains 24 expert strategies across 5 categories:
- Trend-following (6)
- Mean-reversion (5)
- Volatility-based (4)
- Volume-based (3)
- Momentum (3)
- Hybrid (3)

Each strategy implements the ExpertStrategy interface.
"""

from .base import (
    ExpertStrategy,
    BaselineStrategy,
    Signal,
    StrategyConfig,
    StrategyResult,
    BuyAndHoldStrategy,
    SMA200Strategy,
    GoldenCrossBaselineStrategy,
)
from .trend_following import (
    GoldenCrossStrategy,
    MACDTrendStrategy,
    ADXBreakoutStrategy,
    IchimokuStrategy,
    ParabolicSARStrategy,
    DonchianBreakoutStrategy,
)
from .mean_reversion import (
    RSIReversalStrategy,
    BollingerBounceStrategy,
    StochasticStrategy,
    WilliamsRStrategy,
    CCIReversalStrategy,
)
from .volatility import (
    ATRBreakoutStrategy,
    KeltnerBreakoutStrategy,
    VolTargetingStrategy,
    BBSqueezeStrategy,
)
from .volume import (
    OBVConfirmationStrategy,
    MFIReversalStrategy,
    VWAPReversionStrategy,
)
from .momentum import (
    MomentumStrategy,
    AroonTrendStrategy,
    TRIXTrendStrategy,
)

__all__ = [
    # Base classes
    "ExpertStrategy",
    "BaselineStrategy",
    "Signal",
    "StrategyConfig",
    "StrategyResult",
    # Baseline strategies
    "BuyAndHoldStrategy",
    "SMA200Strategy",
    "GoldenCrossBaselineStrategy",
    # Trend-following
    "GoldenCrossStrategy",
    "MACDTrendStrategy",
    "ADXBreakoutStrategy",
    "IchimokuStrategy",
    "ParabolicSARStrategy",
    "DonchianBreakoutStrategy",
    # Mean-reversion
    "RSIReversalStrategy",
    "BollingerBounceStrategy",
    "StochasticStrategy",
    "WilliamsRStrategy",
    "CCIReversalStrategy",
    # Volatility
    "ATRBreakoutStrategy",
    "KeltnerBreakoutStrategy",
    "VolTargetingStrategy",
    "BBSqueezeStrategy",
    # Volume
    "OBVConfirmationStrategy",
    "MFIReversalStrategy",
    "VWAPReversionStrategy",
    # Momentum
    "MomentumStrategy",
    "AroonTrendStrategy",
    "TRIXTrendStrategy",
]

"""
Base Expert Strategy Interface
==============================

All expert strategies inherit from ExpertStrategy and implement
the generate_signal() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import numpy as np


class Signal(Enum):
    """Trading signal types."""
    LONG = 1       # Go long QQQ
    SHORT = -1     # Go short QQQ
    CASH = 0       # Stay in cash


@dataclass
class StrategyConfig:
    """Configuration for an expert strategy."""
    name: str
    category: str  # trend, mean_reversion, volatility, volume, momentum, hybrid
    parameters: dict
    regime_hypothesis: str  # When should this strategy work?
    failure_mode: str  # When does it fail?


@dataclass
class StrategyResult:
    """Result of strategy signal generation."""
    signal: Signal
    confidence: float  # 0.0 to 1.0
    raw_indicator: float  # The underlying indicator value
    metadata: dict  # Additional info


class ExpertStrategy(ABC):
    """
    Base class for all expert trading strategies.

    Each expert strategy:
    1. Has a clear hypothesis about when it works
    2. Generates signals (LONG, SHORT, CASH)
    3. Provides confidence estimates
    4. Documents failure modes
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.name
        self.category = config.category
        self.parameters = config.parameters

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        """
        Generate a trading signal for a given point in time.

        Args:
            data: DataFrame with OHLCV and indicators
            idx: Index position to generate signal for (for walk-forward)

        Returns:
            StrategyResult with signal and metadata

        Note:
            This method must NOT use any data after idx (no look-ahead).
        """
        pass

    def compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute any additional indicators needed by this strategy.

        Override in subclass if needed.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with additional indicator columns
        """
        return data

    def get_position_size(
        self,
        signal: Signal,
        confidence: float,
        max_leverage: float = 3.0,
    ) -> float:
        """
        Determine position size based on signal and confidence.

        Args:
            signal: The trading signal
            confidence: Confidence level (0-1)
            max_leverage: Maximum allowed leverage

        Returns:
            Position size (-3.0 to +3.0)
        """
        if signal == Signal.CASH:
            return 0.0

        # Scale by confidence
        base_size = 1.0 if signal == Signal.LONG else -1.0

        # Higher confidence = larger position (up to max leverage)
        if confidence > 0.8:
            multiplier = max_leverage
        elif confidence > 0.6:
            multiplier = max_leverage * 0.7
        elif confidence > 0.4:
            multiplier = max_leverage * 0.5
        else:
            multiplier = 1.0

        return base_size * multiplier

    def backtest(
        self,
        data: pd.DataFrame,
        start_idx: int = 200,  # Skip warmup period
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV and indicators
            start_idx: Index to start from (allow indicator warmup)

        Returns:
            DataFrame with signals and positions
        """
        results = []

        for idx in range(start_idx, len(data)):
            result = self.generate_signal(data, idx)
            results.append({
                "date": data.index[idx],
                "close": data["close"].iloc[idx],
                "signal": result.signal.value,
                "confidence": result.confidence,
                "raw_indicator": result.raw_indicator,
                "position": self.get_position_size(
                    result.signal, result.confidence
                ),
            })

        return pd.DataFrame(results).set_index("date")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class BaselineStrategy(ExpertStrategy):
    """
    Base class for baseline strategies.

    Baselines are simpler strategies used for comparison.
    They don't need regime awareness - they are the benchmarks.
    """

    def __init__(self, name: str, parameters: dict = None):
        config = StrategyConfig(
            name=name,
            category="baseline",
            parameters=parameters or {},
            regime_hypothesis="Baseline - not regime-dependent",
            failure_mode="N/A - used as benchmark",
        )
        super().__init__(config)

    def get_position_size(
        self,
        signal: Signal,
        confidence: float,
        max_leverage: float = 3.0,
    ) -> float:
        """
        Baseline strategies use 1x leverage (no leverage).
        This ensures fair comparison with buy-and-hold.
        """
        if signal == Signal.CASH:
            return 0.0
        return 1.0 if signal == Signal.LONG else -1.0


class BuyAndHoldStrategy(BaselineStrategy):
    """Always long QQQ baseline."""

    def __init__(self):
        super().__init__("BuyAndHold", {})

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        return StrategyResult(
            signal=Signal.LONG,
            confidence=1.0,
            raw_indicator=1.0,
            metadata={"strategy": "buy_and_hold"},
        )


class SMA200Strategy(BaselineStrategy):
    """Long when price > 200 SMA, else cash."""

    def __init__(self):
        super().__init__("SMA200", {"period": 200})

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        close = data["close"].iloc[idx]
        sma_200 = data["close"].iloc[max(0, idx - 200):idx + 1].mean()

        if close > sma_200:
            signal = Signal.LONG
            confidence = min(1.0, (close / sma_200 - 1) * 10 + 0.5)
        else:
            signal = Signal.CASH
            confidence = min(1.0, (1 - close / sma_200) * 10 + 0.5)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=close / sma_200,
            metadata={"sma_200": sma_200},
        )


class GoldenCrossBaselineStrategy(BaselineStrategy):
    """Long when 50 SMA > 200 SMA."""

    def __init__(self):
        super().__init__("GoldenCross", {"fast": 50, "slow": 200})

    def generate_signal(self, data: pd.DataFrame, idx: int) -> StrategyResult:
        if idx < 200:
            return StrategyResult(Signal.CASH, 0.0, 0.0, {})

        sma_50 = data["close"].iloc[idx - 50:idx + 1].mean()
        sma_200 = data["close"].iloc[idx - 200:idx + 1].mean()

        if sma_50 > sma_200:
            signal = Signal.LONG
            confidence = min(1.0, (sma_50 / sma_200 - 1) * 20 + 0.5)
        else:
            signal = Signal.CASH
            confidence = min(1.0, (1 - sma_50 / sma_200) * 20 + 0.5)

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            raw_indicator=sma_50 / sma_200,
            metadata={"sma_50": sma_50, "sma_200": sma_200},
        )

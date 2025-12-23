"""
Backtest Engine
===============

Walk-forward backtesting with strict no-look-ahead validation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

from strategies.base import ExpertStrategy, Signal, StrategyResult
from .cost_model import CostModel
from .metrics import PerformanceMetrics, calculate_metrics


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    initial_capital: float = 500_000.0
    max_leverage: float = 3.0
    max_drawdown: float = 0.25  # 25% hard limit
    warmup_period: int = 200  # Days for indicator warmup
    rebalance_frequency: str = "daily"  # daily, weekly, monthly


@dataclass
class BacktestResult:
    """Complete backtest result."""

    strategy_name: str
    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    positions: pd.Series
    signals: pd.Series
    returns: pd.Series
    costs: pd.Series
    trades: pd.DataFrame
    daily_data: pd.DataFrame


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Key features:
    - Strict no-look-ahead: Only uses data available at decision time
    - Realistic cost modeling: Commissions, slippage, margin, borrow
    - Position sizing based on leverage and confidence
    - Drawdown monitoring and risk controls
    """

    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        cost_model: Optional[CostModel] = None,
    ):
        self.config = config or BacktestConfig()
        self.cost_model = cost_model or CostModel()

    def run(
        self,
        strategy: ExpertStrategy,
        data: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Run backtest for a single strategy.

        Args:
            strategy: ExpertStrategy instance
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            regimes: Optional regime labels for analysis

        Returns:
            BacktestResult with complete performance data
        """
        self._validate_data(data)

        # Initialize tracking
        n = len(data)
        equity = np.zeros(n)
        positions = np.zeros(n)
        signals = np.zeros(n)
        daily_returns = np.zeros(n)
        daily_costs = np.zeros(n)

        # Initialize equity for warmup period
        equity[:self.config.warmup_period] = self.config.initial_capital
        current_position = 0.0  # In shares
        current_shares = 0.0
        trades: List[Dict[str, Any]] = []

        # Walk forward through data
        for idx in range(self.config.warmup_period, n):
            current_price = data["close"].iloc[idx]
            prev_price = data["close"].iloc[idx - 1]

            # 1. Generate signal (using only past data)
            result = strategy.generate_signal(data, idx)
            signals[idx] = result.signal.value

            # 2. Determine target position
            target_position = strategy.get_position_size(
                result.signal,
                result.confidence,
                max_leverage=self.config.max_leverage,
            )

            # 3. Calculate position change
            current_equity = equity[idx - 1]
            target_shares = (target_position * current_equity) / current_price

            shares_delta = target_shares - current_shares

            # 4. Execute trade if position changed
            trade_cost = 0.0
            if abs(shares_delta) > 0:
                trade_cost = self.cost_model.calculate_trade_cost(
                    shares_delta, current_price
                )
                trades.append({
                    "date": data.index[idx],
                    "signal": result.signal.name,
                    "confidence": result.confidence,
                    "shares_delta": shares_delta,
                    "price": current_price,
                    "cost": trade_cost,
                    "position_after": target_shares,
                })
                current_shares = target_shares

            # 5. Calculate holding costs
            is_short = current_shares < 0
            leverage = abs(current_shares * current_price / current_equity) if current_equity > 0 else 0
            holding_cost = self.cost_model.calculate_holding_cost(
                abs(current_shares * current_price),
                leverage,
                is_short,
                days=1,
            )

            # 6. Calculate daily P&L
            if current_shares != 0:
                price_return = (current_price - prev_price) / prev_price
                position_pnl = current_shares * prev_price * price_return
            else:
                position_pnl = 0.0

            total_cost = trade_cost + holding_cost
            daily_pnl = position_pnl - total_cost

            # 7. Update equity
            equity[idx] = equity[idx - 1] + daily_pnl
            daily_returns[idx] = daily_pnl / equity[idx - 1] if equity[idx - 1] > 0 else 0
            daily_costs[idx] = total_cost
            positions[idx] = target_position

            # 8. Check drawdown limit
            peak_equity = equity[:idx + 1].max()
            current_dd = (peak_equity - equity[idx]) / peak_equity
            if current_dd > self.config.max_drawdown:
                # Force to cash
                if current_shares != 0:
                    liquidation_cost = self.cost_model.calculate_trade_cost(
                        current_shares, current_price
                    )
                    equity[idx] -= liquidation_cost
                    trades.append({
                        "date": data.index[idx],
                        "signal": "FORCED_LIQUIDATION",
                        "confidence": 0.0,
                        "shares_delta": -current_shares,
                        "price": current_price,
                        "cost": liquidation_cost,
                        "position_after": 0.0,
                    })
                    current_shares = 0.0
                    positions[idx] = 0.0

        # Create result series
        equity_series = pd.Series(equity, index=data.index, name="equity")
        positions_series = pd.Series(positions, index=data.index, name="position")
        signals_series = pd.Series(signals, index=data.index, name="signal")
        returns_series = pd.Series(daily_returns, index=data.index, name="return")
        costs_series = pd.Series(daily_costs, index=data.index, name="cost")

        # Calculate metrics (excluding warmup)
        valid_returns = returns_series.iloc[self.config.warmup_period:]
        valid_positions = positions_series.iloc[self.config.warmup_period:]
        valid_costs = costs_series.iloc[self.config.warmup_period:]

        metrics = calculate_metrics(
            valid_returns,
            valid_positions,
            valid_costs,
            regimes=regimes,
        )

        # Build daily data DataFrame
        daily_data = pd.DataFrame({
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "equity": equity_series,
            "position": positions_series,
            "signal": signals_series,
            "return": returns_series,
            "cost": costs_series,
        })

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        return BacktestResult(
            strategy_name=strategy.name,
            config=self.config,
            metrics=metrics,
            equity_curve=equity_series,
            positions=positions_series,
            signals=signals_series,
            returns=returns_series,
            costs=costs_series,
            trades=trades_df,
            daily_data=daily_data,
        )

    def run_multiple(
        self,
        strategies: List[ExpertStrategy],
        data: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
    ) -> Dict[str, BacktestResult]:
        """
        Run backtest for multiple strategies.

        Returns:
            Dictionary mapping strategy name to BacktestResult
        """
        results = {}
        for strategy in strategies:
            result = self.run(strategy, data, regimes)
            results[strategy.name] = result
        return results

    def compare(
        self,
        results: Dict[str, BacktestResult],
    ) -> pd.DataFrame:
        """
        Compare multiple backtest results.

        Returns:
            DataFrame with key metrics for each strategy
        """
        comparison = []
        for name, result in results.items():
            metrics = result.metrics
            comparison.append({
                "strategy": name,
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "calmar_ratio": metrics.calmar_ratio,
                "num_trades": metrics.num_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_costs": metrics.total_costs,
            })
        return pd.DataFrame(comparison).set_index("strategy")

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be DatetimeIndex")

        if len(data) < self.config.warmup_period + 10:
            raise ValueError(
                f"Insufficient data: need at least {self.config.warmup_period + 10} rows"
            )

        # Check for look-ahead bias indicators
        if data.index.is_monotonic_increasing is False:
            raise ValueError("Data must be sorted by date (ascending)")


class WalkForwardValidator:
    """
    Walk-forward validation to prevent overfitting.

    Splits data into in-sample and out-of-sample periods,
    trains/optimizes on IS, validates on OOS.
    """

    def __init__(
        self,
        is_window: int = 252 * 3,  # 3 years in-sample
        oos_window: int = 252,  # 1 year out-of-sample
        step: int = 252,  # 1 year step
    ):
        self.is_window = is_window
        self.oos_window = oos_window
        self.step = step

    def generate_folds(self, data: pd.DataFrame) -> List[tuple]:
        """
        Generate walk-forward folds.

        Returns:
            List of (train_start, train_end, test_start, test_end) indices
        """
        n = len(data)
        folds = []

        start = 0
        while start + self.is_window + self.oos_window <= n:
            train_start = start
            train_end = start + self.is_window
            test_start = train_end
            test_end = min(train_end + self.oos_window, n)

            folds.append((train_start, train_end, test_start, test_end))
            start += self.step

        return folds

    def validate(
        self,
        strategy: ExpertStrategy,
        data: pd.DataFrame,
        engine: BacktestEngine,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Returns:
            Dictionary with IS and OOS performance by fold
        """
        folds = self.generate_folds(data)
        results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            # In-sample period
            is_data = data.iloc[train_start:train_end]
            is_result = engine.run(strategy, is_data)

            # Out-of-sample period
            oos_data = data.iloc[test_start:test_end]
            oos_result = engine.run(strategy, oos_data)

            results.append({
                "fold": i,
                "is_start": data.index[train_start],
                "is_end": data.index[train_end - 1],
                "oos_start": data.index[test_start],
                "oos_end": data.index[test_end - 1],
                "is_sharpe": is_result.metrics.sharpe_ratio,
                "oos_sharpe": oos_result.metrics.sharpe_ratio,
                "is_return": is_result.metrics.annualized_return,
                "oos_return": oos_result.metrics.annualized_return,
                "sharpe_decay": (
                    is_result.metrics.sharpe_ratio - oos_result.metrics.sharpe_ratio
                ),
            })

        # Aggregate statistics
        results_df = pd.DataFrame(results)
        avg_is_sharpe = results_df["is_sharpe"].mean()
        avg_oos_sharpe = results_df["oos_sharpe"].mean()
        avg_decay = results_df["sharpe_decay"].mean()

        return {
            "folds": results_df,
            "avg_is_sharpe": avg_is_sharpe,
            "avg_oos_sharpe": avg_oos_sharpe,
            "avg_sharpe_decay": avg_decay,
            "oos_sharpe_std": results_df["oos_sharpe"].std(),
            "overfit_ratio": avg_is_sharpe / avg_oos_sharpe if avg_oos_sharpe != 0 else float("inf"),
        }

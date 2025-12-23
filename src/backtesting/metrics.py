"""
Performance Metrics
===================

Comprehensive performance metrics for strategy evaluation.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class PerformanceMetrics:
    """
    Complete performance metrics for a backtest.
    """

    # Returns
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float

    # Risk
    max_drawdown: float
    max_drawdown_duration: int  # days
    calmar_ratio: float
    var_95: float  # 95% Value at Risk (daily)
    cvar_95: float  # Conditional VaR

    # Trading
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_holding_period: float  # days

    # Costs
    total_costs: float
    gross_return: float
    net_return: float

    # Regime performance
    bull_return: Optional[float] = None
    bear_return: Optional[float] = None
    sideways_return: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "calmar_ratio": self.calmar_ratio,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "avg_holding_period": self.avg_holding_period,
            "total_costs": self.total_costs,
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "bull_return": self.bull_return,
            "bear_return": self.bear_return,
            "sideways_return": self.sideways_return,
        }


def calculate_metrics(
    returns: pd.Series,
    positions: pd.Series,
    costs: pd.Series,
    risk_free_rate: float = 0.04,
    regimes: Optional[pd.Series] = None,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Daily strategy returns (net of costs)
        positions: Daily position sizes
        costs: Daily trading/holding costs
        risk_free_rate: Annual risk-free rate
        regimes: Optional regime labels for conditional analysis

    Returns:
        PerformanceMetrics object
    """
    # Clean data
    returns = returns.dropna()
    if len(returns) == 0:
        raise ValueError("No valid returns to analyze")

    # Basic stats
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annualized_vol = returns.std() * np.sqrt(252)

    # Risk-adjusted returns
    excess_return = annualized_return - risk_free_rate
    sharpe = excess_return / annualized_vol if annualized_vol > 0 else 0

    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = excess_return / downside_std if downside_std > 0 else 0

    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Max drawdown duration
    dd_duration = 0
    max_dd_duration = 0
    for dd in drawdown:
        if dd < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

    # VaR and CVaR
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    # Trade analysis
    position_changes = positions.diff().fillna(0)
    trades = position_changes[position_changes != 0]
    num_trades = len(trades)

    # Win/loss analysis (by holding period)
    trade_returns = []
    current_entry = None
    current_return = 0

    for i in range(1, len(positions)):
        if positions.iloc[i] != positions.iloc[i - 1]:
            if current_entry is not None:
                trade_returns.append(current_return)
            current_entry = i
            current_return = 0
        else:
            current_return += returns.iloc[i]

    if current_entry is not None and current_return != 0:
        trade_returns.append(current_return)

    trade_returns = pd.Series(trade_returns) if trade_returns else pd.Series([0])
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    win_rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

    # Average holding period
    holding_periods = []
    current_holding = 0
    for i in range(1, len(positions)):
        if positions.iloc[i] != 0:
            current_holding += 1
        elif current_holding > 0:
            holding_periods.append(current_holding)
            current_holding = 0
    if current_holding > 0:
        holding_periods.append(current_holding)
    avg_holding = np.mean(holding_periods) if holding_periods else 0

    # Costs
    total_costs = costs.sum()
    gross_returns = returns + costs  # Add back costs for gross
    gross_return = (1 + gross_returns).prod() - 1

    # Regime analysis
    bull_return = None
    bear_return = None
    sideways_return = None

    if regimes is not None:
        regimes = regimes.reindex(returns.index)
        for regime in regimes.unique():
            if pd.isna(regime):
                continue
            regime_mask = regimes == regime
            regime_returns = returns[regime_mask]
            if len(regime_returns) > 0:
                regime_ann_ret = (1 + regime_returns).prod() ** (252 / len(regime_returns)) - 1
                if "BULL" in str(regime).upper():
                    bull_return = regime_ann_ret
                elif "BEAR" in str(regime).upper():
                    bear_return = regime_ann_ret
                else:
                    sideways_return = regime_ann_ret

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        calmar_ratio=calmar,
        var_95=var_95,
        cvar_95=cvar_95,
        num_trades=num_trades,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        avg_holding_period=avg_holding,
        total_costs=total_costs,
        gross_return=gross_return,
        net_return=total_return,
        bull_return=bull_return,
        bear_return=bear_return,
        sideways_return=sideways_return,
    )

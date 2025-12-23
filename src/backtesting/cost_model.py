"""
Cost Model for Backtesting
==========================

Realistic transaction cost modeling including:
- Commission (per-share or per-trade)
- Slippage (market impact)
- Margin interest (for leverage)
- Borrow costs (for shorting)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CostModel:
    """
    Transaction and holding cost model.

    Based on Strategy Charter §9 cost model.
    """

    # Trading costs (per trade)
    commission_per_share: float = 0.005  # $0.005/share
    min_commission: float = 1.0  # $1 minimum per trade
    slippage_bps: float = 2.0  # 2 bps market impact

    # Holding costs (annualized rates)
    margin_interest_rate: float = 0.07  # 7% annual for margin
    borrow_rate: float = 0.005  # 0.5% annual for QQQ (easy to borrow)

    # Position limits
    max_leverage: float = 3.0

    def calculate_trade_cost(
        self,
        shares: float,
        price: float,
        is_entry: bool = True,
    ) -> float:
        """
        Calculate total cost of a trade.

        Args:
            shares: Number of shares traded (absolute value)
            price: Price per share
            is_entry: True if entering position, False if exiting

        Returns:
            Total cost in dollars
        """
        shares = abs(shares)
        notional = shares * price

        # Commission
        commission = max(shares * self.commission_per_share, self.min_commission)

        # Slippage (market impact)
        slippage = notional * (self.slippage_bps / 10000)

        return commission + slippage

    def calculate_holding_cost(
        self,
        position_value: float,
        leverage: float,
        is_short: bool,
        days: int = 1,
    ) -> float:
        """
        Calculate daily holding costs.

        Args:
            position_value: Absolute value of position
            leverage: Current leverage ratio
            is_short: True if short position
            days: Number of days held

        Returns:
            Holding cost in dollars
        """
        daily_margin_rate = self.margin_interest_rate / 252
        daily_borrow_rate = self.borrow_rate / 252

        cost = 0.0

        # Margin interest on leveraged portion
        if leverage > 1.0:
            margin_amount = position_value * (leverage - 1.0) / leverage
            cost += margin_amount * daily_margin_rate * days

        # Borrow cost for shorts
        if is_short:
            cost += position_value * daily_borrow_rate * days

        return cost

    def calculate_round_trip_cost(
        self,
        shares: float,
        entry_price: float,
        exit_price: float,
        holding_days: int,
        leverage: float = 1.0,
        is_short: bool = False,
    ) -> dict:
        """
        Calculate total round-trip cost.

        Returns:
            Dictionary with cost breakdown
        """
        position_value = abs(shares * entry_price)

        entry_cost = self.calculate_trade_cost(shares, entry_price, is_entry=True)
        exit_cost = self.calculate_trade_cost(shares, exit_price, is_entry=False)
        holding_cost = self.calculate_holding_cost(
            position_value, leverage, is_short, holding_days
        )

        total = entry_cost + exit_cost + holding_cost

        return {
            "entry_cost": entry_cost,
            "exit_cost": exit_cost,
            "holding_cost": holding_cost,
            "total_cost": total,
            "cost_bps": (total / position_value) * 10000 if position_value > 0 else 0,
        }

    def adjust_return_for_costs(
        self,
        gross_return: float,
        turnover: float,
        avg_leverage: float = 1.0,
        short_ratio: float = 0.0,
        holding_period_days: float = 21,
    ) -> float:
        """
        Adjust gross return for estimated costs.

        Args:
            gross_return: Gross return (e.g., 0.10 for 10%)
            turnover: Annual turnover (e.g., 12 for monthly rebalance)
            avg_leverage: Average leverage used
            short_ratio: Fraction of time in short positions
            holding_period_days: Average holding period in days

        Returns:
            Net return after costs
        """
        # Trading costs (per round trip)
        trade_cost_bps = 2 * self.slippage_bps + (self.commission_per_share * 100)
        annual_trade_cost = turnover * (trade_cost_bps / 10000)

        # Margin interest
        if avg_leverage > 1.0:
            margin_ratio = (avg_leverage - 1.0) / avg_leverage
            margin_cost = margin_ratio * self.margin_interest_rate
        else:
            margin_cost = 0.0

        # Borrow cost
        borrow_cost = short_ratio * self.borrow_rate

        total_cost = annual_trade_cost + margin_cost + borrow_cost
        net_return = gross_return - total_cost

        return net_return

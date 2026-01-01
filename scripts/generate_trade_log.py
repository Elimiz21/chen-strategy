#!/usr/bin/env python3
"""
Generate Comprehensive Trade Log
=================================

Creates a detailed CSV file with all trades over the entire period,
including position changes, signals from all 7 strategies, micro-regime
states, and strategy weight adjustments.
"""

import sys
import os

# Add both project root and src to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pandas as pd
import numpy as np

# Import project modules
from regime.micro_regimes import MicroRegimeDetector


# ============================================================================
# Load REAL QQQ Data (from cached CSV)
# ============================================================================

def load_real_qqq_data():
    """Load real QQQ data from cached CSV file."""
    # Use the correct cached data file with real prices
    data_file = os.path.join(project_root, "data", "qqq_2000-01-01_2024-12-31.csv")

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Real QQQ data not found at {data_file}")

    print(f"Loading real QQQ data from {data_file}")
    df = pd.read_csv(data_file, parse_dates=["date"], index_col="date")

    # Standardize column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Add returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

    return df


# ============================================================================
# Cost Model
# ============================================================================

class CostModel:
    """Transaction and holding cost model."""
    commission_per_share = 0.005
    min_commission = 1.0
    slippage_bps = 2.0

    def calculate_trade_cost(self, shares, price):
        shares = abs(shares)
        notional = shares * price
        commission = max(shares * self.commission_per_share, self.min_commission)
        slippage = notional * (self.slippage_bps / 10000)
        return commission + slippage


# ============================================================================
# Strategy Implementations
# ============================================================================

class BBSqueezeStrategy:
    """Bollinger Band Squeeze Strategy."""
    name = "BBSqueeze"

    def __init__(self, bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult

    def generate_signal(self, data, idx):
        if idx < max(self.bb_period, self.kc_period) + 5:
            return {"signal": 0, "confidence": 0, "reason": "Insufficient data"}

        close = data["close"].iloc[:idx + 1]
        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]

        # Bollinger Bands
        sma = close.iloc[-self.bb_period:].mean()
        std = close.iloc[-self.bb_period:].std()
        bb_upper = sma + self.bb_std * std
        bb_lower = sma - self.bb_std * std
        bb_width = (bb_upper - bb_lower) / sma

        # Keltner Channels (using ATR approximation)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.iloc[-self.kc_period:].mean()
        kc_upper = sma + self.kc_mult * atr
        kc_lower = sma - self.kc_mult * atr

        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]

        # Squeeze: BB inside KC
        squeeze = bb_upper < kc_upper and bb_lower > kc_lower

        # Breakout from squeeze
        momentum = (current_price - close.iloc[-self.bb_period:].mean()) / std if std > 0 else 0

        if squeeze:
            return {"signal": 0, "confidence": 0.3, "reason": f"In squeeze, BB width={bb_width:.4f}"}
        elif momentum > 1.5 and current_price > prev_price:
            return {"signal": 1, "confidence": min(0.9, momentum / 3), "reason": f"Breakout UP, momentum={momentum:.2f}"}
        elif momentum < -1.5 and current_price < prev_price:
            return {"signal": -1, "confidence": min(0.9, abs(momentum) / 3), "reason": f"Breakout DOWN, momentum={momentum:.2f}"}
        else:
            return {"signal": 0, "confidence": 0.2, "reason": f"No clear signal, momentum={momentum:.2f}"}


class DonchianBreakoutStrategy:
    """Donchian Channel Breakout Strategy."""
    name = "DonchianBreakout"

    def __init__(self, period=20):
        self.period = period

    def generate_signal(self, data, idx):
        if idx < self.period + 1:
            return {"signal": 0, "confidence": 0, "reason": "Insufficient data"}

        high = data["high"].iloc[idx - self.period:idx]
        low = data["low"].iloc[idx - self.period:idx]
        close = data["close"].iloc[idx]

        upper = high.max()
        lower = low.min()
        mid = (upper + lower) / 2

        if close > upper:
            dist = (close - upper) / (upper - mid) if upper != mid else 1
            return {"signal": 1, "confidence": min(0.9, 0.5 + dist * 0.3),
                    "reason": f"Break above {self.period}d high=${upper:.2f}"}
        elif close < lower:
            dist = (lower - close) / (mid - lower) if mid != lower else 1
            return {"signal": -1, "confidence": min(0.9, 0.5 + dist * 0.3),
                    "reason": f"Break below {self.period}d low=${lower:.2f}"}
        else:
            position = (close - lower) / (upper - lower) if upper != lower else 0.5
            return {"signal": 0, "confidence": 0.3,
                    "reason": f"Within channel, position={position:.1%}"}


class KeltnerBreakoutStrategy:
    """Keltner Channel Breakout Strategy."""
    name = "KeltnerBreakout"

    def __init__(self, period=20, mult=2.0):
        self.period = period
        self.mult = mult

    def generate_signal(self, data, idx):
        if idx < self.period + 5:
            return {"signal": 0, "confidence": 0, "reason": "Insufficient data"}

        close = data["close"].iloc[:idx + 1]
        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]

        # EMA
        ema = close.ewm(span=self.period, adjust=False).mean().iloc[-1]

        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.iloc[-self.period:].mean()

        upper = ema + self.mult * atr
        lower = ema - self.mult * atr
        current = close.iloc[-1]

        if current > upper:
            return {"signal": 1, "confidence": 0.7,
                    "reason": f"Above Keltner upper=${upper:.2f}, ATR=${atr:.2f}"}
        elif current < lower:
            return {"signal": -1, "confidence": 0.7,
                    "reason": f"Below Keltner lower=${lower:.2f}, ATR=${atr:.2f}"}
        else:
            return {"signal": 0, "confidence": 0.3,
                    "reason": f"Within Keltner channel, EMA=${ema:.2f}"}


class IchimokuStrategy:
    """Ichimoku Cloud Strategy."""
    name = "Ichimoku"

    def __init__(self, tenkan=9, kijun=26, senkou_b=52):
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_b = senkou_b

    def generate_signal(self, data, idx):
        if idx < self.senkou_b + 26:
            return {"signal": 0, "confidence": 0, "reason": "Insufficient data"}

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        # Tenkan-sen (Conversion Line)
        tenkan = (high.iloc[-self.tenkan:].max() + low.iloc[-self.tenkan:].min()) / 2

        # Kijun-sen (Base Line)
        kijun = (high.iloc[-self.kijun:].max() + low.iloc[-self.kijun:].min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = (tenkan + kijun) / 2

        # Senkou Span B (Leading Span B)
        senkou_b = (high.iloc[-self.senkou_b:].max() + low.iloc[-self.senkou_b:].min()) / 2

        current = close.iloc[-1]
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Signal logic
        if current > cloud_top and tenkan > kijun:
            return {"signal": 1, "confidence": 0.8,
                    "reason": f"Above cloud, TK cross bullish, Tenkan=${tenkan:.2f}"}
        elif current < cloud_bottom and tenkan < kijun:
            return {"signal": -1, "confidence": 0.8,
                    "reason": f"Below cloud, TK cross bearish, Tenkan=${tenkan:.2f}"}
        elif current > cloud_top:
            return {"signal": 1, "confidence": 0.5,
                    "reason": f"Above cloud, price=${current:.2f}"}
        elif current < cloud_bottom:
            return {"signal": -1, "confidence": 0.5,
                    "reason": f"Below cloud, price=${current:.2f}"}
        else:
            return {"signal": 0, "confidence": 0.2,
                    "reason": f"Inside cloud, range=${cloud_bottom:.2f}-${cloud_top:.2f}"}


class ParabolicSARStrategy:
    """Parabolic SAR Strategy."""
    name = "ParabolicSAR"

    def __init__(self, af_start=0.02, af_increment=0.02, af_max=0.2):
        self.af_start = af_start
        self.af_increment = af_increment
        self.af_max = af_max

    def generate_signal(self, data, idx):
        if idx < 20:
            return {"signal": 0, "confidence": 0, "reason": "Insufficient data"}

        high = data["high"].iloc[:idx + 1]
        low = data["low"].iloc[:idx + 1]
        close = data["close"].iloc[:idx + 1]

        # Simplified PSAR calculation
        sar = low.iloc[-20:-10].min()
        ep = high.iloc[-10:].max()
        af = self.af_start
        uptrend = True

        for i in range(-10, 0):
            if uptrend:
                sar = sar + af * (ep - sar)
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + self.af_increment, self.af_max)
                if low.iloc[i] < sar:
                    uptrend = False
                    sar = ep
                    ep = low.iloc[i]
                    af = self.af_start
            else:
                sar = sar - af * (sar - ep)
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + self.af_increment, self.af_max)
                if high.iloc[i] > sar:
                    uptrend = True
                    sar = ep
                    ep = high.iloc[i]
                    af = self.af_start

        current = close.iloc[-1]

        if uptrend and current > sar:
            return {"signal": 1, "confidence": 0.6,
                    "reason": f"Uptrend, SAR=${sar:.2f}, EP=${ep:.2f}"}
        elif not uptrend and current < sar:
            return {"signal": -1, "confidence": 0.6,
                    "reason": f"Downtrend, SAR=${sar:.2f}, EP=${ep:.2f}"}
        else:
            return {"signal": 0, "confidence": 0.3,
                    "reason": f"SAR reversal pending, SAR=${sar:.2f}"}


class TrendEnsembleStrategy:
    """Trend Ensemble Strategy - Multi-lookback trend signals."""
    name = "TrendEnsemble"

    def __init__(self, lookbacks=(10, 20, 50, 100, 200)):
        self.lookbacks = lookbacks

    def generate_signal(self, data, idx):
        max_lookback = max(self.lookbacks)
        if idx < max_lookback + 1:
            return {"signal": 0, "confidence": 0, "reason": "Insufficient data"}

        close = data["close"].iloc[:idx + 1]

        signals = []
        for lookback in self.lookbacks:
            if idx >= lookback:
                ret = (close.iloc[-1] / close.iloc[-lookback]) - 1
                normalized_ret = ret * np.sqrt(20 / lookback)
                signals.append(np.sign(normalized_ret))
            else:
                signals.append(0)

        avg_signal = np.mean(signals)
        agreement = abs(avg_signal)

        if avg_signal > 0.2:
            return {"signal": 1, "confidence": agreement,
                    "reason": f"Trend UP, avg_signal={avg_signal:.2f}, agreement={agreement:.2f}"}
        elif avg_signal < -0.2:
            return {"signal": -1, "confidence": agreement,
                    "reason": f"Trend DOWN, avg_signal={avg_signal:.2f}, agreement={agreement:.2f}"}
        else:
            return {"signal": 0, "confidence": 0.3,
                    "reason": f"No trend, avg_signal={avg_signal:.2f}"}


class ROROStrategy:
    """Risk-On/Risk-Off Strategy."""
    name = "RORO"

    def __init__(self, vol_threshold=0.25, ma_periods=(20, 50, 200), momentum_periods=(5, 20, 60)):
        self.vol_threshold = vol_threshold
        self.ma_periods = ma_periods
        self.momentum_periods = momentum_periods

    def generate_signal(self, data, idx):
        max_period = max(max(self.ma_periods), max(self.momentum_periods))
        if idx < max_period + 1:
            return {"signal": 0, "confidence": 0, "reason": "Insufficient data"}

        close = data["close"].iloc[:idx + 1]
        current_price = close.iloc[-1]

        risk_score = 0
        max_score = 0

        # Volatility check
        returns = close.pct_change().iloc[-20:]
        current_vol = returns.std() * np.sqrt(252)
        if current_vol > self.vol_threshold:
            risk_score += 2
        max_score += 2

        # MA checks
        for period in self.ma_periods:
            ma = close.iloc[-period:].mean()
            if current_price < ma:
                risk_score += 1
            max_score += 1

        # Momentum checks
        neg_momentum_count = 0
        for period in self.momentum_periods:
            ret = (current_price / close.iloc[-period]) - 1
            if ret < 0:
                neg_momentum_count += 1

        if neg_momentum_count >= 2:
            risk_score += 2
        max_score += 2

        risk_off_pct = risk_score / max_score

        if risk_off_pct > 0.6:
            return {"signal": 0, "confidence": risk_off_pct,
                    "reason": f"RISK-OFF, score={risk_score}/{max_score}, vol={current_vol:.1%}"}
        elif risk_off_pct > 0.4:
            return {"signal": 1, "confidence": 0.3,
                    "reason": f"Cautious, score={risk_score}/{max_score}"}
        else:
            return {"signal": 1, "confidence": 1.0 - risk_off_pct,
                    "reason": f"RISK-ON, score={risk_score}/{max_score}"}


# ============================================================================
# Main Trade Log Generator
# ============================================================================

def generate_trade_log():
    """Generate comprehensive trade log CSV."""
    print("=" * 70)
    print("GENERATING COMPREHENSIVE TRADE LOG")
    print("=" * 70)

    # Load REAL data
    print("\n1. Loading REAL QQQ data...")
    data = load_real_qqq_data()

    # Verify prices are correct
    print(f"\nPrice verification:")
    print(f"  2000 start: ${data['close'].iloc[0]:.2f}")
    print(f"  2024 end:   ${data['close'].iloc[-1]:.2f}")
    print(f"  Max price:  ${data['close'].max():.2f}")

    if data['close'].max() < 200:
        raise ValueError("ERROR: Prices look wrong! Max should be >$500 for 2024")

    # Initialize components
    print("\n2. Initializing strategies and regime detector...")
    regime_detector = MicroRegimeDetector()
    cost_model = CostModel()

    # Initialize all 7 strategies
    strategies = {
        "BBSqueeze": BBSqueezeStrategy(),
        "DonchianBreakout": DonchianBreakoutStrategy(),
        "KeltnerBreakout": KeltnerBreakoutStrategy(),
        "Ichimoku": IchimokuStrategy(),
        "ParabolicSAR": ParabolicSARStrategy(),
        "TrendEnsemble": TrendEnsembleStrategy(),
        "RORO": ROROStrategy(),
    }

    # Base weights
    base_weights = {
        "BBSqueeze": 0.25,
        "DonchianBreakout": 0.25,
        "KeltnerBreakout": 0.15,
        "Ichimoku": 0.10,
        "ParabolicSAR": 0.05,
        "TrendEnsemble": 0.10,
        "RORO": 0.10,
    }

    # Simulation parameters
    initial_capital = 500_000
    warmup = 200

    print("\n3. Running portfolio simulation...")

    # Track state
    equity = initial_capital
    peak_equity = initial_capital
    position = 0.0  # -2 to +2 (leverage)
    current_weights = base_weights.copy()
    trades = []

    for idx in range(warmup, len(data)):
        if idx % 500 == 0:
            print(f"   Processing day {idx}/{len(data)} - {data.index[idx].date()} - Price: ${data['close'].iloc[idx]:.2f}...")

        date = data.index[idx]
        current_price = data["close"].iloc[idx]
        prev_price = data["close"].iloc[idx - 1]

        # Calculate daily P&L from existing position FIRST
        if position != 0:
            price_return = (current_price - prev_price) / prev_price
            daily_pnl = position * equity * price_return
            equity += daily_pnl
        else:
            daily_pnl = 0

        # Update peak
        if equity > peak_equity:
            peak_equity = equity

        # Detect regime
        regime = regime_detector.detect(data, idx)

        # Get signals from all strategies
        signals = {}
        for name, strategy in strategies.items():
            try:
                sig = strategy.generate_signal(data, idx)
                signals[name] = sig
            except Exception as e:
                signals[name] = {"signal": 0, "confidence": 0, "reason": f"Error: {str(e)[:50]}"}

        # Calculate weighted signal
        weighted_signal = 0
        total_weight = 0
        for name, sig in signals.items():
            weight = current_weights.get(name, 0)
            weighted_signal += sig["signal"] * sig["confidence"] * weight
            total_weight += weight

        if total_weight > 0:
            weighted_signal /= total_weight

        # Determine target position
        if weighted_signal > 0.3:
            target_position = min(2.0, weighted_signal * 2)
        elif weighted_signal < -0.3:
            target_position = max(-2.0, weighted_signal * 2)
        else:
            target_position = 0

        # Regime-based weight adjustments
        new_weights = base_weights.copy()
        weight_changes = []

        if regime:
            # Crisis volatility
            if regime.volatility.name == "CRISIS":
                for name in new_weights:
                    if name != "RORO":
                        old = new_weights[name]
                        new_weights[name] *= 0.7
                        if abs(new_weights[name] - old) > 0.01:
                            weight_changes.append(f"{name}: {old:.0%}->{new_weights[name]:.0%}")
                old_roro = new_weights["RORO"]
                new_weights["RORO"] = min(0.35, new_weights["RORO"] + 0.15)
                weight_changes.append(f"RORO: {old_roro:.0%}->{new_weights['RORO']:.0%} [CRISIS]")
                target_position *= 0.5

            elif regime.volatility.name == "HIGH":
                for name in new_weights:
                    if name != "RORO":
                        new_weights[name] *= 0.85
                new_weights["RORO"] = min(0.25, new_weights["RORO"] + 0.05)
                target_position *= 0.75

            # Strong bear
            if regime.trend.name == "STRONG_BEAR":
                for name in ["DonchianBreakout", "Ichimoku", "ParabolicSAR"]:
                    old = new_weights[name]
                    new_weights[name] *= 0.8
                    if abs(new_weights[name] - old) > 0.01:
                        weight_changes.append(f"{name}: {old:.0%}->{new_weights[name]:.0%} [BEAR]")
                old_roro = new_weights["RORO"]
                new_weights["RORO"] = min(0.35, new_weights["RORO"] + 0.10)
                if abs(new_weights["RORO"] - old_roro) > 0.01:
                    weight_changes.append(f"RORO: {old_roro:.0%}->{new_weights['RORO']:.0%}")

        # Normalize weights
        total_w = sum(new_weights.values())
        if total_w > 0:
            new_weights = {k: v/total_w for k, v in new_weights.items()}

        # Drawdown-based adjustments
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
        if current_dd > 0.15:
            target_position *= 0.5
            weight_changes.append(f"DD={current_dd:.1%}: pos*0.5")
        elif current_dd > 0.10:
            target_position *= 0.75
            weight_changes.append(f"DD={current_dd:.1%}: pos*0.75")

        # Check if we have a trade (position change > 5%)
        position_change = target_position - position
        is_trade = abs(position_change) > 0.05

        if is_trade:
            # Calculate trade details
            trade_value = abs(position_change) * equity
            shares = trade_value / current_price
            trade_cost = cost_model.calculate_trade_cost(shares, current_price)

            # Create trade record
            trade_record = {
                "date": date.strftime("%Y-%m-%d"),
                "price": round(current_price, 2),
                "prev_position": round(position, 3),
                "new_position": round(target_position, 3),
                "position_change": round(position_change, 3),
                "trade_value_usd": round(trade_value, 2),
                "trade_cost_usd": round(trade_cost, 2),
                "equity_before": round(equity, 2),
                "daily_pnl": round(daily_pnl, 2),
                "equity_after_trade": round(equity - trade_cost, 2),
                "current_dd_pct": round(current_dd * 100, 2),
                "peak_equity": round(peak_equity, 2),

                # Regime info
                "regime_trend": regime.trend.name if regime else "N/A",
                "regime_volatility": regime.volatility.name if regime else "N/A",
                "regime_momentum": regime.momentum.name if regime else "N/A",
                "regime_mean_rev": regime.mean_reversion.name if regime else "N/A",
                "regime_code": regime.code if regime else "N/A",

                # Weighted signal
                "weighted_signal": round(weighted_signal, 3),

                # Individual strategy signals and reasons
                "BBSqueeze_signal": signals["BBSqueeze"]["signal"],
                "BBSqueeze_conf": round(signals["BBSqueeze"]["confidence"], 2),
                "BBSqueeze_reason": signals["BBSqueeze"]["reason"],

                "DonchianBreakout_signal": signals["DonchianBreakout"]["signal"],
                "DonchianBreakout_conf": round(signals["DonchianBreakout"]["confidence"], 2),
                "DonchianBreakout_reason": signals["DonchianBreakout"]["reason"],

                "KeltnerBreakout_signal": signals["KeltnerBreakout"]["signal"],
                "KeltnerBreakout_conf": round(signals["KeltnerBreakout"]["confidence"], 2),
                "KeltnerBreakout_reason": signals["KeltnerBreakout"]["reason"],

                "Ichimoku_signal": signals["Ichimoku"]["signal"],
                "Ichimoku_conf": round(signals["Ichimoku"]["confidence"], 2),
                "Ichimoku_reason": signals["Ichimoku"]["reason"],

                "ParabolicSAR_signal": signals["ParabolicSAR"]["signal"],
                "ParabolicSAR_conf": round(signals["ParabolicSAR"]["confidence"], 2),
                "ParabolicSAR_reason": signals["ParabolicSAR"]["reason"],

                "TrendEnsemble_signal": signals["TrendEnsemble"]["signal"],
                "TrendEnsemble_conf": round(signals["TrendEnsemble"]["confidence"], 2),
                "TrendEnsemble_reason": signals["TrendEnsemble"]["reason"],

                "RORO_signal": signals["RORO"]["signal"],
                "RORO_conf": round(signals["RORO"]["confidence"], 2),
                "RORO_reason": signals["RORO"]["reason"],

                # Weight changes
                "weight_changes": "; ".join(weight_changes) if weight_changes else "None",

                # Current weights
                "w_BBSqueeze": round(new_weights["BBSqueeze"], 3),
                "w_DonchianBreakout": round(new_weights["DonchianBreakout"], 3),
                "w_KeltnerBreakout": round(new_weights["KeltnerBreakout"], 3),
                "w_Ichimoku": round(new_weights["Ichimoku"], 3),
                "w_ParabolicSAR": round(new_weights["ParabolicSAR"], 3),
                "w_TrendEnsemble": round(new_weights["TrendEnsemble"], 3),
                "w_RORO": round(new_weights["RORO"], 3),
            }

            trades.append(trade_record)

            # Apply trade cost to equity
            equity -= trade_cost

            # Update position
            position = target_position

        current_weights = new_weights

    print(f"\n4. Found {len(trades)} trades")

    # Create DataFrame
    df = pd.DataFrame(trades)

    # Save to CSV
    output_path = os.path.join(project_root, "results", "comprehensive_trade_log.csv")
    df.to_csv(output_path, index=False)
    print(f"\n5. Saved to: {output_path}")

    # Summary stats
    print("\n" + "=" * 70)
    print("TRADE LOG SUMMARY")
    print("=" * 70)
    print(f"Total trades: {len(trades)}")
    print(f"Date range: {trades[0]['date']} to {trades[-1]['date']}")
    print(f"Average trades per year: {len(trades) / 25:.1f}")

    # Price verification
    print(f"\nPrice samples from trade log:")
    early_trades = [t for t in trades if t['date'] < '2005-01-01']
    late_trades = [t for t in trades if t['date'] > '2023-01-01']
    if early_trades:
        print(f"  Early (2000-2004): ${early_trades[0]['price']:.2f} - ${early_trades[-1]['price']:.2f}")
    if late_trades:
        print(f"  Late (2023-2024):  ${late_trades[0]['price']:.2f} - ${late_trades[-1]['price']:.2f}")

    # Position distribution
    long_trades = sum(1 for t in trades if t["new_position"] > 0)
    short_trades = sum(1 for t in trades if t["new_position"] < 0)
    flat_trades = sum(1 for t in trades if t["new_position"] == 0)
    print(f"\nPosition distribution:")
    print(f"  Long entries:  {long_trades} ({long_trades/len(trades)*100:.1f}%)")
    print(f"  Short entries: {short_trades} ({short_trades/len(trades)*100:.1f}%)")
    print(f"  Flat entries:  {flat_trades} ({flat_trades/len(trades)*100:.1f}%)")

    # Regime distribution
    print(f"\nRegime at trade time:")
    regime_counts = df["regime_trend"].value_counts()
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} ({count/len(trades)*100:.1f}%)")

    # Equity evolution
    print(f"\nEquity evolution:")
    print(f"  Start:  ${trades[0]['equity_before']:,.2f}")
    print(f"  End:    ${trades[-1]['equity_after_trade']:,.2f}")

    print(f"\nFile size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Columns: {len(df.columns)}")

    return df


if __name__ == "__main__":
    generate_trade_log()

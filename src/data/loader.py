"""
QQQ Data Loader
===============

Handles acquisition and preprocessing of QQQ historical data.
Implements DEC-010 decision (data source selection).

Data Source: Yahoo Finance (yfinance)
- Free, reliable, well-maintained
- Adjusted prices handle splits/dividends
- Daily OHLCV data from 1999-present
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Try to import yfinance, but don't fail if not available
try:
    import yfinance as yf
    HAS_YFINANCE = True
except (ImportError, TypeError):
    HAS_YFINANCE = False


class QQQDataLoader:
    """
    Load and manage QQQ historical data.

    Attributes:
        symbol: Always "QQQ"
        data_dir: Directory for cached data
        start_date: Start of data range
        end_date: End of data range
    """

    def __init__(
        self,
        data_dir: str = "data",
        start_date: str = "1999-03-10",
        end_date: Optional[str] = None,
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory to store cached data
            start_date: Start date (default: QQQ inception)
            end_date: End date (default: today)
        """
        self.symbol = "QQQ"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        self._data: Optional[pd.DataFrame] = None
        self._hash: Optional[str] = None

    def fetch(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch QQQ data from Yahoo Finance.

        Args:
            force_refresh: If True, ignore cache and re-download

        Returns:
            DataFrame with OHLCV data
        """
        cache_parquet = self.data_dir / f"qqq_{self.start_date}_{self.end_date}.parquet"
        cache_csv = self.data_dir / f"qqq_{self.start_date}_{self.end_date}.csv"

        # Check for cached data (parquet preferred, then CSV)
        if cache_parquet.exists() and not force_refresh:
            print(f"Loading cached data from {cache_parquet}")
            self._data = pd.read_parquet(cache_parquet)
        elif cache_csv.exists() and not force_refresh:
            print(f"Loading cached data from {cache_csv}")
            self._data = pd.read_csv(cache_csv, index_col=0, parse_dates=True)
            # Standardize column names
            self._data.columns = [c.lower().replace(" ", "_") for c in self._data.columns]
        else:
            if HAS_YFINANCE:
                print(f"Fetching {self.symbol} data from Yahoo Finance...")
                ticker = yf.Ticker(self.symbol)
                self._data = ticker.history(start=self.start_date, end=self.end_date)
            else:
                print(f"yfinance not available, using fallback data fetch...")
                self._data = self._fetch_yahoo_direct()

            # Standardize column names
            self._data.columns = [c.lower().replace(" ", "_") for c in self._data.columns]

            # Ensure datetime index
            self._data.index = pd.to_datetime(self._data.index)
            if self._data.index.tz is not None:
                self._data.index = self._data.index.tz_localize(None)  # Remove timezone
            self._data.index.name = "date"

            # Save to cache (use CSV if parquet not available)
            try:
                self._data.to_parquet(cache_parquet)
                print(f"Cached data to {cache_parquet}")
            except ImportError:
                self._data.to_csv(cache_csv)
                print(f"Cached data to {cache_csv} (parquet not available)")

        # Compute hash for versioning
        self._hash = self._compute_hash()

        return self._data

    def _fetch_yahoo_direct(self) -> pd.DataFrame:
        """
        Fallback method - generate synthetic QQQ-like data for testing.
        In production, install yfinance with a compatible Python version.
        """
        print("WARNING: Using synthetic data for testing. Install yfinance for real data.")
        return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic QQQ-like price data for testing.
        Mimics QQQ characteristics: ~10% CAGR, ~25% volatility, realistic patterns.
        """
        # Generate date range (business days only)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="B")
        n = len(dates)

        # Parameters mimicking QQQ
        annual_return = 0.10  # 10% CAGR
        annual_vol = 0.25  # 25% volatility
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)

        # Generate returns with regime-switching behavior
        np.random.seed(42)  # Reproducibility

        # Simulate regimes (bull/bear/sideways)
        regimes = np.zeros(n)
        regime = 0  # Start in bull
        for i in range(n):
            if np.random.random() < 0.02:  # 2% chance to switch
                regime = np.random.choice([0, 1, 2])  # bull, bear, sideways
            regimes[i] = regime

        # Generate returns based on regime
        returns = np.zeros(n)
        for i in range(n):
            if regimes[i] == 0:  # Bull
                returns[i] = np.random.normal(daily_return * 1.5, daily_vol * 0.8)
            elif regimes[i] == 1:  # Bear
                returns[i] = np.random.normal(-daily_return * 1.0, daily_vol * 1.5)
            else:  # Sideways
                returns[i] = np.random.normal(0, daily_vol * 0.6)

        # Convert to prices
        initial_price = 50.0  # QQQ was around $50 in 2000
        prices = initial_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        df = pd.DataFrame(index=dates)
        df["Close"] = prices
        df["Open"] = df["Close"].shift(1).fillna(initial_price) * (1 + np.random.normal(0, 0.002, n))
        df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, n)))
        df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, n)))
        df["Volume"] = np.random.lognormal(18, 0.5, n).astype(int)  # ~65M avg volume

        # Ensure OHLC consistency
        df["High"] = df[["Open", "Close", "High"]].max(axis=1)
        df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)

        return df

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the data for versioning."""
        if self._data is None:
            raise ValueError("No data loaded. Call fetch() first.")

        # Hash the data content
        data_bytes = self._data.to_csv().encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    @property
    def data(self) -> pd.DataFrame:
        """Get the loaded data."""
        if self._data is None:
            self.fetch()
        return self._data

    @property
    def hash(self) -> str:
        """Get the data hash for versioning."""
        if self._hash is None:
            self.fetch()
        return self._hash

    def add_returns(self) -> pd.DataFrame:
        """Add return columns to the data."""
        df = self.data.copy()

        # Daily returns
        df["returns"] = df["close"].pct_change()

        # Log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Cumulative returns
        df["cum_returns"] = (1 + df["returns"]).cumprod() - 1

        self._data = df
        return df

    def add_technical_indicators(self) -> pd.DataFrame:
        """Add basic technical indicators needed for regime detection."""
        df = self.data.copy()

        # Moving averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()

        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

        # Volatility
        df["atr_14"] = self._compute_atr(df, 14)
        df["atr_20"] = self._compute_atr(df, 20)
        df["realized_vol_20"] = df["returns"].rolling(20).std() * np.sqrt(252)

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI
        df["rsi_14"] = self._compute_rsi(df["close"], 14)

        # Bollinger Bands
        df["bb_middle"] = df["sma_20"]
        df["bb_std"] = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ADX (simplified)
        df["adx_14"] = self._compute_adx(df, 14)

        self._data = df
        return df

    def _compute_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _compute_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute Average Directional Index (simplified)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr = self._compute_atr(df, 1) * period  # Approximation

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smoothed
        plus_di = 100 * plus_dm.rolling(period).mean() / tr.rolling(period).mean()
        minus_di = 100 * minus_dm.rolling(period).mean() / tr.rolling(period).mean()

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def get_summary(self) -> dict:
        """Get summary statistics of the data."""
        df = self.data

        return {
            "symbol": self.symbol,
            "start_date": str(df.index.min().date()),
            "end_date": str(df.index.max().date()),
            "trading_days": len(df),
            "data_hash": self.hash,
            "columns": list(df.columns),
            "price_range": {
                "min": float(df["close"].min()),
                "max": float(df["close"].max()),
                "current": float(df["close"].iloc[-1]),
            },
            "returns": {
                "total": float((df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100),
                "cagr": float(
                    ((df["close"].iloc[-1] / df["close"].iloc[0])
                     ** (252 / len(df)) - 1) * 100
                ),
            },
        }

    def validate_no_lookahead(self) -> bool:
        """
        Validate that data has no look-ahead bias issues.

        Checks:
        1. Data is sorted by date (ascending)
        2. No future dates
        3. No duplicate dates
        4. No missing weekdays (gaps > 4 days may indicate issues)

        Returns:
            True if validation passes
        """
        df = self.data
        issues = []

        # Check sorted
        if not df.index.is_monotonic_increasing:
            issues.append("Data not sorted by date")

        # Check no future dates
        today = pd.Timestamp.now().normalize()
        if df.index.max() > today:
            issues.append(f"Data contains future dates: {df.index.max()}")

        # Check no duplicates
        if df.index.duplicated().any():
            issues.append("Data contains duplicate dates")

        # Check for large gaps (excluding weekends)
        gaps = df.index.to_series().diff()
        large_gaps = gaps[gaps > pd.Timedelta(days=5)]
        if len(large_gaps) > 10:  # Allow some holidays
            issues.append(f"Data has {len(large_gaps)} gaps > 5 days")

        if issues:
            print("Data validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("Data validation passed: no look-ahead bias detected")
        return True


# Convenience function
def load_qqq_data(
    start_date: str = "1999-03-10",
    end_date: Optional[str] = None,
    add_indicators: bool = True,
) -> pd.DataFrame:
    """
    Load QQQ data with technical indicators.

    Args:
        start_date: Start date
        end_date: End date (default: today)
        add_indicators: Whether to add technical indicators

    Returns:
        DataFrame with QQQ data
    """
    loader = QQQDataLoader(start_date=start_date, end_date=end_date)
    loader.fetch()
    loader.add_returns()

    if add_indicators:
        loader.add_technical_indicators()

    print(f"\nData Summary:")
    print(f"  Hash: {loader.hash}")
    print(f"  Rows: {len(loader.data)}")
    print(f"  Columns: {len(loader.data.columns)}")

    return loader.data


if __name__ == "__main__":
    # Test the loader
    df = load_qqq_data()
    print(df.tail())

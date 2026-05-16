"""Feature engineering module for stock analysis."""

import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of prices
        period: RSI period (default: 14)
    
    Returns:
        Series with RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series of prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
    
    Returns:
        Tuple of (MACD, Signal Line) Series
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced stock indicators to the dataframe.
    
    Features include:
    - Returns: pct_change, log_return
    - Moving averages: MA7, MA14, MA30, EMA12, EMA26
    - Momentum: RSI14, MACD, MACD_Signal
    - Volatility: rolling_std14, hl_spread
    - Volume: volume_change, volume_ma
    - Target: price_up (1 if next day price increases, else 0)
    
    Args:
        df: Stock data DataFrame with columns [Open, High, Low, Close, Volume]
    
    Returns:
        DataFrame with engineered features, NaN rows dropped
    """
    df = df.copy()
    
    # 1. RETURNS
    df['pct_change'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. MOVING AVERAGES
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA14'] = df['Close'].rolling(window=14).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # 3. MOMENTUM
    df['RSI14'] = calculate_rsi(df['Close'], period=14)
    macd, macd_signal = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    
    # 4. VOLATILITY
    df['rolling_std14'] = df['Close'].rolling(window=14).std()
    # Handle division by zero for hl_spread
    df['hl_spread'] = np.where(
        df['Close'] != 0,
        (df['High'] - df['Low']) / df['Close'],
        0
    )
    
    # 5. VOLUME
    df['volume_change'] = df['Volume'].pct_change()
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    
    # 6. TARGET VARIABLE
    # 1 if next day price increases, 0 otherwise
    df['price_up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    # Drop NaN rows
    df = df.dropna()
    
    return df


def calculate_moving_averages(df: pd.DataFrame, windows: list = [20, 50, 200]) -> pd.DataFrame:
    """
    Calculate moving averages for stock prices.
    
    Args:
        df: Stock data DataFrame
        windows: List of window sizes for moving averages
    
    Returns:
        DataFrame with moving average features added
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    out = df.copy()
    for window in windows:
        out[f"MA{window}"] = out['Close'].rolling(window=window).mean()
    return out


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate rolling volatility.
    
    Args:
        df: Stock data DataFrame
        window: Rolling window size
    
    Returns:
        DataFrame with volatility feature added
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    out = df.copy()
    out[f"volatility_{window}"] = out['Close'].pct_change().rolling(window=window).std()
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all features for model training.
    
    Args:
        df: Stock data DataFrame
    
    Returns:
        DataFrame with all engineered features
    """
    return add_features(df)

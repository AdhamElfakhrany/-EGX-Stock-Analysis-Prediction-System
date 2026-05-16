"""Utility functions for stock analysis."""

import pandas as pd
from datetime import datetime, timedelta


def get_date_range(days_back: int = 365) -> tuple:
    """
    Get date range for data fetching.
    
    Args:
        days_back: Number of days to go back from today
    
    Returns:
        Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
    """
    if days_back <= 0:
        raise ValueError("days_back must be a positive integer")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def validate_ticker(ticker: str) -> bool:
    """
    Validate stock ticker symbol.
    
    Args:
        ticker: Ticker symbol to validate
    
    Returns:
        Boolean indicating if ticker is valid
    """
    if not isinstance(ticker, str):
        return False
    cleaned = ticker.strip().upper()
    if not cleaned:
        return False
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    return all(char in allowed_chars for char in cleaned)


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate daily returns from prices.
    
    Args:
        prices: Series of stock prices
    
    Returns:
        Series of daily returns
    """
    if prices is None or prices.empty:
        raise ValueError("prices series cannot be empty")
    return prices.pct_change()


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format value as currency string.
    
    Args:
        value: Numerical value
        decimals: Number of decimal places
    
    Returns:
        Formatted currency string
    """
    if decimals < 0:
        raise ValueError("decimals must be >= 0")
    return f"${value:,.{decimals}f}"

"""ETL (Extract, Transform, Load) module for stock data."""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

STOCKS = {
    "CIB": "COMI.CA",
    "FAISAL": "FAIT.CA",
    "HDB": "HDBK.CA",
    "ADIB": "ADIB.CA",
}


def fetch_data(ticker: str = "COMI.CA") -> pd.DataFrame:
    """
    Fetch historical stock data for the last 5 years.
    
    Args:
        ticker: Stock ticker symbol (default: 'COMI.CA')
    
    Returns:
        DataFrame with OHLCV data (Open, High, Low, Close, Volume)
    """
    # Calculate date range: last 5 years
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=5*365)
    
    df = fetch_stock_data(
        ticker=ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    df = clean_stock_data(df)
    
    # Print basic info
    print(f"\n{'='*50}")
    print(f"Stock Data: {ticker}")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Date Range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"{'='*50}\n")
    
    return df


def validate_ticker(stock_name: str, ticker: str) -> Optional[str]:
    """
    Validate ticker by attempting to fetch a small recent window.

    Returns:
        Working ticker symbol or None if unavailable.
    """
    candidates = [ticker]
    for candidate in candidates:
        try:
            probe = yf.download(candidate, period="1mo", progress=False, auto_adjust=False)
            if not probe.empty:
                logging.info(f"[{stock_name}] Using ticker {candidate}")
                return candidate
        except Exception as exc:
            logging.warning(f"[{stock_name}] Ticker probe failed for {candidate}: {exc}")
    logging.error(f"Ticker {stock_name} failed or returned no data")
    return None


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in format 'YYYY-MM-DD'
        end_date: End date in format 'YYYY-MM-DD'
    
    Returns:
        DataFrame with OHLCV data
    """
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol must be a non-empty string")

    symbol = ticker.strip().upper()
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
    except Exception as exc:
        logging.warning(f"Date-based download failed for {symbol}: {exc}")
        df = pd.DataFrame()

    if df.empty:
        # Fallback path avoids local DST edge-cases in explicit datetime windows.
        df = yf.download(
            symbol,
            period="5y",
            progress=False,
            auto_adjust=False
        )
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' between {start_date} and {end_date}"
        )
    return df


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess stock data.
    
    Args:
        df: Raw stock data DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")

    df = df.copy()

    # Flatten MultiIndex columns if Yahoo returns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in fetched data: {missing}")

    df = df[required_columns].sort_index()
    df = df.dropna()
    return df


def save_stock_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save stock data to CSV file.
    
    Args:
        df: Stock data DataFrame
        filepath: Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df.to_csv(filepath)
    logging.info(f"Data saved to {filepath} - shape: {df.shape}")


def _resolve_stock_name(stock_name: Optional[str], ticker: str) -> Optional[str]:
    """Resolve stock name from explicit input or STOCKS reverse lookup."""
    if stock_name:
        return stock_name.upper()
    for name, symbol in STOCKS.items():
        if symbol.upper() == ticker.upper():
            return name
    return None


def add_news_sentiment_features(
    df: pd.DataFrame, ticker: str, stock_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Add latest-row sentiment features from Kafka stream, fallback to API.
    """
    from src.news_fetcher import fetch_news
    from src.sentiment import analyze_sentiment

    out = df.copy()
    out["avg_sentiment"] = 0.0
    out["news_count"] = 0
    out["positive_news"] = 0
    out["negative_news"] = 0

    if out.empty:
        return out

    resolved_name = _resolve_stock_name(stock_name, ticker)
    stream_data = None
    stream_path = "data/news_stream.csv"
    
    if os.path.exists(stream_path):
        try:
            stream_df = pd.read_csv(stream_path)
            if not stream_df.empty and "stock" in stream_df.columns:
                stock_df = stream_df[stream_df["stock"] == resolved_name]
                if not stock_df.empty:
                    avg_sentiment = float(stock_df["sentiment"].mean())
                    news_count = len(stock_df)
                    positive_news = len(stock_df[stock_df["sentiment"] > 0])
                    negative_news = len(stock_df[stock_df["sentiment"] < 0])
                    stream_data = {
                        "avg_sentiment": avg_sentiment,
                        "count": news_count,
                        "positive": positive_news,
                        "negative": negative_news,
                    }
                    logging.info(f"Loaded sentiment from Kafka stream for {resolved_name}")
        except Exception as e:
            logging.error(f"Failed to read Kafka stream data from {stream_path}: {e}")

    sentiment = stream_data
    if not sentiment:
        logging.info(
            f"Step 3: Fallback fetching news for sentiment. Stock: {resolved_name or 'UNKNOWN'}"
        )
        news_list = fetch_news(resolved_name or ticker)
        sentiment = analyze_sentiment(news_list)

    latest_index = out.index[-1]
    out.at[latest_index, "avg_sentiment"] = float(sentiment.get("avg_sentiment", 0.0))
    out.at[latest_index, "news_count"] = int(sentiment.get("count", 0))
    out.at[latest_index, "positive_news"] = int(sentiment.get("positive", 0))
    out.at[latest_index, "negative_news"] = int(sentiment.get("negative", 0))

    out["avg_sentiment"] = out["avg_sentiment"].fillna(0.0).astype(float)
    out["news_count"] = out["news_count"].fillna(0).astype(int)
    out["positive_news"] = out["positive_news"].fillna(0).astype(int)
    out["negative_news"] = out["negative_news"].fillna(0).astype(int)

    logging.info(
        "Sentiment summary applied to latest row: avg_sentiment=%.4f, news_count=%d, positive=%d, negative=%d",
        sentiment.get("avg_sentiment", 0.0),
        sentiment.get("count", 0),
        sentiment.get("positive", 0),
        sentiment.get("negative", 0),
    )
    return out


def run_pipeline(
    ticker: str = "COMI.CA",
    output_path: str = "data/processed_stock.csv",
    stock_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the complete ETL pipeline: fetch -> engineer features -> save.
    
    Args:
        ticker: Stock ticker symbol (default: 'COMI.CA')
        output_path: Path to save processed data (default: 'data/processed_stock.csv')
        stock_name: Optional display stock name for bank-specific news query
    
    Returns:
        Processed DataFrame with features
    """
    logging.info(f"Starting ETL pipeline for {ticker}")
    
    # Step 1: Fetch data
    logging.info(f"Step 1: Fetching historical data for {ticker}...")
    df_raw = fetch_data(ticker)
    logging.info(f"Fetched {df_raw.shape[0]} records")
    
    # Step 2: Apply feature engineering
    logging.info("Step 2: Engineering features...")
    from src.features import add_features
    df_features = add_features(df_raw)
    logging.info(f"Created {df_features.shape[1]} features")
    logging.info(f"Final dataset shape: {df_features.shape}")
    
    # Step 3: Add sentiment features from news
    df_features = add_news_sentiment_features(df_features, ticker=ticker, stock_name=stock_name)
    logging.info(f"Final dataset shape after sentiment merge: {df_features.shape}")

    # Step 4: Save to CSV
    logging.info(f"Step 4: Saving processed data to {output_path}...")
    save_stock_data(df_features, output_path)
    
    logging.info(f"Output columns: {list(df_features.columns)}")
    logging.info("ETL pipeline completed successfully!")
    return df_features


def run_multi_stock_pipeline(
    stocks: Dict[str, str] = STOCKS,
    output_dir: str = "data",
    also_save_default: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run ETL pipeline for multiple stocks and save each stock to a separate file.

    Args:
        stocks: Mapping of stock display name -> ticker symbol
        output_dir: Directory where processed CSV files are saved
        also_save_default: If True, save CIB output to data/processed_stock.csv for compatibility

    Returns:
        Dictionary of stock display name -> processed DataFrame
    """
    outputs: Dict[str, pd.DataFrame] = {}
    failed_stocks: Dict[str, str] = {}
    logging.info(f"Starting multi-stock ETL for {len(stocks)} stocks")

    for stock_name, ticker in stocks.items():
        stock_output_path = os.path.join(output_dir, f"{stock_name}_processed.csv")
        logging.info(f"[{stock_name}] Running pipeline for ticker {ticker}")
        try:
            valid_ticker = validate_ticker(stock_name, ticker)
            if valid_ticker is None:
                failed_stocks[stock_name] = "Ticker unavailable"
                continue

            df_processed = run_pipeline(
                ticker=valid_ticker,
                output_path=stock_output_path,
                stock_name=stock_name
            )
            outputs[stock_name] = df_processed
            logging.info(
                f"[{stock_name}] Completed successfully. Rows fetched: {df_processed.shape[0]}. "
                f"Saved to: {stock_output_path}"
            )

            # Keep existing dashboard default path working without changes.
            if also_save_default and stock_name == "CIB":
                save_stock_data(df_processed, os.path.join(output_dir, "processed_stock.csv"))
                logging.info("[CIB] Also saved to data/processed_stock.csv (backward compatibility)")
        except Exception as exc:
            failed_stocks[stock_name] = str(exc)
            logging.error(f"[{stock_name}] Failed: {exc}")
            continue

    logging.info(
        f"Multi-stock ETL completed. Successful: {len(outputs)} | Failed: {len(failed_stocks)}"
    )
    if failed_stocks:
        logging.warning(f"Failed stock details: {failed_stocks}")

    if not outputs:
        raise RuntimeError("Multi-stock ETL failed for all configured stocks")

    return outputs


if __name__ == "__main__":
    # Run the multi-stock pipeline while keeping single-stock compatibility.
    try:
        all_processed = run_multi_stock_pipeline()
        
        print("\n" + "="*60)
        print("PIPELINE OUTPUT SUMMARY")
        print("="*60)
        for stock_name, df_processed in all_processed.items():
            print(f"\n[{stock_name}] Dataset shape: {df_processed.shape}")
            print(f"Columns ({len(df_processed.columns)}):")
            print(", ".join(df_processed.columns))
        print("="*60)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


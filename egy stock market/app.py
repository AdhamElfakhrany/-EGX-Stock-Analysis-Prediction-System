"""Main Streamlit application for stock analysis."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import time
from datetime import datetime, timedelta
import logging
from src.model import StockPricePredictor, predict_latest
from src.etl import STOCKS
from src.news_fetcher import fetch_news_for_bank
from src.sentiment import analyze_sentiment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SENTIMENT_COLUMNS = ["avg_sentiment", "news_count", "positive_news", "negative_news"]


# ============ SESSION STATE INITIALIZATION ============
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 10


@st.cache_resource(show_spinner=False)
def train_cached_model(df: pd.DataFrame) -> StockPricePredictor:
    """Train and cache ML model from processed dataset."""
    model = StockPricePredictor()
    X = df.drop(columns=['price_up'], errors='ignore')
    y = df['price_up']
    model.train(X, y)
    return model


@st.cache_resource(show_spinner=False)
def train_cached_model_variant(df: pd.DataFrame, include_sentiment: bool) -> StockPricePredictor:
    """Train model variant with or without sentiment features."""
    model = StockPricePredictor()
    X = df.drop(columns=["price_up"], errors="ignore").copy()
    if not include_sentiment:
        X = X.drop(columns=SENTIMENT_COLUMNS, errors="ignore")
    y = df["price_up"]
    model.train(X, y)
    return model


def render_prediction_panel(df_filtered: pd.DataFrame, stock_name: str):
    """Render latest prediction and confidence values."""
    try:
        if 'price_up' not in df_filtered.columns:
            st.warning("Prediction unavailable: target column `price_up` not found.")
            return

        model = train_cached_model(df_filtered)
        pred = predict_latest(df_filtered, model)

        st.markdown("---")
        st.subheader("🤖 Next Day Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal", pred["prediction"])
        with col2:
            st.metric("Confidence", f"{pred['confidence'] * 100:.2f}%")
        with col3:
            st.metric("Latest Close", f"${pred['latest_price']:.2f}")

        st.caption(
            f"Probabilities -> UP: {pred['probability_up'] * 100:.2f}% | "
            f"DOWN: {pred['probability_down'] * 100:.2f}%"
        )
        if {"avg_sentiment", "positive_news", "negative_news"}.issubset(df_filtered.columns):
            latest = df_filtered.iloc[-1]
            st.info(
                "Prediction influenced by sentiment | "
                f"Avg: {latest['avg_sentiment']:.3f}, "
                f"Positive: {int(latest['positive_news'])}, "
                f"Negative: {int(latest['negative_news'])}."
            )
        else:
            st.info("Prediction influenced by sentiment signals when news is available.")

        st.markdown("---")
        st.subheader("🧠 Model Insights")
        metrics = getattr(model, "last_metrics", None)
        if not metrics:
            st.info("Model insights are temporarily unavailable.")
            return

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with m2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        with m3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with m4:
            st.metric("Prediction Confidence", f"{pred['confidence'] * 100:.2f}%")

        cm = metrics.get("confusion_matrix")
        if cm is not None:
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Predicted DOWN", "Predicted UP"],
                    y=["Actual DOWN", "Actual UP"],
                    colorscale="Blues",
                    showscale=True,
                    text=cm,
                    texttemplate="%{text}",
                )
            )
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=350,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        latest_features = (
            df_filtered.iloc[[-1]]
            .drop(columns=["price_up"], errors="ignore")
            .T.rename(columns={df_filtered.index[-1]: "latest_value"})
        )
        with st.expander("Latest features used for prediction", expanded=False):
            st.dataframe(latest_features, use_container_width=True)

        st.markdown("---")
        st.subheader("⚖️ With Sentiment vs Without Sentiment")
        try:
            model_with_sentiment = train_cached_model_variant(df_filtered, include_sentiment=True)
            model_without_sentiment = train_cached_model_variant(df_filtered, include_sentiment=False)
            pred_with = predict_latest(df_filtered, model_with_sentiment)
            pred_without = predict_latest(df_filtered, model_without_sentiment)
            metrics_with = model_with_sentiment.last_metrics or {}
            metrics_without = model_without_sentiment.last_metrics or {}

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**With Sentiment**")
                st.metric("Accuracy", f"{metrics_with.get('accuracy', 0.0):.3f}")
                st.metric("Prediction", pred_with["prediction"])
                st.metric("Confidence", f"{pred_with['confidence'] * 100:.2f}%")
            with c2:
                st.markdown("**Without Sentiment**")
                st.metric("Accuracy", f"{metrics_without.get('accuracy', 0.0):.3f}")
                st.metric("Prediction", pred_without["prediction"])
                st.metric("Confidence", f"{pred_without['confidence'] * 100:.2f}%")

            accuracy_diff = metrics_with.get("accuracy", 0.0) - metrics_without.get("accuracy", 0.0)
            st.caption(
                f"Accuracy delta (With - Without): {accuracy_diff:+.4f}. "
                "Use this to evaluate whether news sentiment improves prediction quality."
            )
        except Exception as comp_exc:
            logger.exception("Model comparison failed")
            st.warning(f"Model comparison unavailable: {comp_exc}")
    except Exception as exc:
        logger.exception("Failed to generate prediction panel")
        st.warning(f"Prediction temporarily unavailable: {exc}")


@st.cache_data(ttl=300)
def load_processed_data_cached(data_path: str = "data/processed_stock.csv"):
    """Load processed stock data with caching."""
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        return df
    return None


def load_processed_data(data_path: str = "data/processed_stock.csv", force_reload: bool = False):
    """Load processed data."""
    if force_reload:
        st.cache_data.clear()
    
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
    return None


def run_etl_pipeline(stock_name: str):
    """Run ETL pipeline."""
    try:
        from src.etl import run_pipeline
        ticker = STOCKS.get(stock_name, "COMI.CA")
        output_path = f"data/{stock_name}_processed.csv"
        logger.info("Starting ETL pipeline...")
        df_processed = run_pipeline(ticker=ticker, output_path=output_path, stock_name=stock_name)
        if stock_name == "CIB":
            df_processed.to_csv("data/processed_stock.csv")
        logger.info(f"ETL completed. Shape: {df_processed.shape}")
        return True
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        return False


@st.cache_data(ttl=300)
def load_latest_news(stock_name: str, limit: int = 5):
    """Fetch latest news items for selected stock."""
    news = fetch_news_for_bank(stock_name)
    return news[:limit]


def render_news_and_sentiment_panel(stock_name: str):
    """Render latest news and sentiment summary for the selected stock."""
    st.markdown("---")
    st.subheader("📰 News & Sentiment")
    try:
        news_items = load_latest_news(stock_name)
        sentiment = analyze_sentiment(news_items)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Avg Sentiment", f"{sentiment['avg_sentiment']:.3f}")
        with c2:
            st.metric("Positive News", f"{sentiment['positive']}")
        with c3:
            st.metric("Negative News", f"{sentiment['negative']}")
        with c4:
            st.metric("Total News", f"{sentiment['count']}")

        if not news_items:
            st.info("No recent news found or NewsAPI key is missing.")
            return

        st.caption(f"Latest headlines for {stock_name}")
        for article in news_items:
            title = article.get("title") or "Untitled article"
            source = article.get("source") or "Unknown source"
            published = article.get("publishedAt") or "Unknown time"
            description = article.get("description") or "No description available."
            st.markdown(f"**{title}**")
            st.caption(f"{source} | {published}")
            st.write(description)
            st.markdown("---")
    except Exception as exc:
        logger.exception("Failed to render news/sentiment panel")
        st.warning(f"News and sentiment panel temporarily unavailable: {exc}")


def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """Create price chart with moving averages."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], mode='lines', name='Close Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Close</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA7'], mode='lines', name='MA7',
        line=dict(color='#ff7f0e', width=1, dash='dash'),
        hovertemplate='<b>MA7</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA30'], mode='lines', name='MA30',
        line=dict(color='#2ca02c', width=1, dash='dash'),
        hovertemplate='<b>MA30</b><br>Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Stock Price with Moving Averages', xaxis_title='Date', yaxis_title='Price ($)',
        hovermode='x unified', template='plotly_white', height=500,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """Create RSI chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI14'], mode='lines', name='RSI (14)',
        line=dict(color='#d62728', width=2),
        hovertemplate='<b>RSI</b><br>Date: %{x|%Y-%m-%d}<br>RSI: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="right")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="right")
    
    fig.update_layout(
        title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI',
        hovermode='x unified', template='plotly_white', height=400,
        yaxis=dict(range=[0, 100]), margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_macd_chart(df: pd.DataFrame) -> go.Figure:
    """Create MACD chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'], mode='lines', name='MACD',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>MACD</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='<b>Signal</b><br>Date: %{x|%Y-%m-%d}<br>Value: %{y:.4f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    
    fig.update_layout(
        title='MACD (Moving Average Convergence Divergence)', xaxis_title='Date', yaxis_title='MACD Value',
        hovermode='x unified', template='plotly_white', height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Create volume chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker=dict(color='#9467bd', opacity=0.7),
        hovertemplate='<b>Volume</b><br>Date: %{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['volume_ma'], mode='lines', name='Volume MA (20)',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='<b>Volume MA</b><br>Date: %{x|%Y-%m-%d}<br>Volume: %{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Trading Volume', xaxis_title='Date', yaxis_title='Volume',
        hovermode='x unified', template='plotly_white', height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def render_dashboard(df_filtered: pd.DataFrame, stock_name: str):
    """Render main dashboard."""
    st.markdown("---")
    st.subheader("📈 Key Metrics")
    
    latest_price = df_filtered['Close'].iloc[-1]
    previous_price = df_filtered['Close'].iloc[-2] if len(df_filtered) > 1 else latest_price
    daily_change = latest_price - previous_price
    daily_change_pct = (daily_change / previous_price * 100) if previous_price != 0 else 0
    latest_volume = df_filtered['Volume'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${latest_price:.2f}", f"{daily_change:.2f}")
    with col2:
        st.metric("Daily Change %", f"{daily_change_pct:.2f}%", delta_color="inverse")
    with col3:
        st.metric("Volume", f"{latest_volume:,.0f}")
    with col4:
        st.metric("Data Points", f"{len(df_filtered)}")
    
    st.markdown("---")
    st.subheader("💹 Price Chart with Moving Averages")
    st.plotly_chart(create_price_chart(df_filtered), use_container_width=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 RSI Indicator")
        st.plotly_chart(create_rsi_chart(df_filtered), use_container_width=True)
    with col2:
        st.subheader("📊 MACD Indicator")
        st.plotly_chart(create_macd_chart(df_filtered), use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Trading Volume")
    st.plotly_chart(create_volume_chart(df_filtered), use_container_width=True)
    render_prediction_panel(df_filtered, stock_name)
    render_news_and_sentiment_panel(stock_name)


def main():
    """Run the Streamlit application."""
    st.set_page_config(page_title="EGX Stock Analysis", layout="wide")
    
    st.title("📊 EGX Stock Analysis")
    st.markdown("Real-time stock analysis dashboard with technical + news sentiment indicators")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.markdown("---")
        st.subheader("🏦 Stock Selector")
        stock_options = list(STOCKS.keys())
        selected_stock = st.selectbox("Choose Bank Stock", stock_options, index=0)
        selected_ticker = STOCKS[selected_stock]
        st.caption(f"Ticker: {selected_ticker}")
        st.markdown("---")
        
        st.subheader("🔄 Real-Time Updates")
        st.session_state.auto_refresh = st.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh)
        
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.slider("Refresh Interval (seconds)", min_value=5, max_value=60, value=st.session_state.refresh_interval)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Now"):
                    with st.spinner("Running ETL pipeline..."):
                        if run_etl_pipeline(selected_stock):
                            st.session_state.last_update = datetime.now()
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ ETL pipeline failed")
            with col2:
                if st.button("⏸️ Stop Auto-Refresh"):
                    st.session_state.auto_refresh = False
                    st.rerun()
        else:
            if st.button("▶️ Start Auto-Refresh"):
                st.session_state.auto_refresh = True
                st.rerun()
        
        st.markdown("---")
        st.subheader("📥 Manual Update")
        if st.button("🔄 Run ETL Pipeline Now"):
            with st.spinner("Running ETL pipeline..."):
                if run_etl_pipeline(selected_stock):
                    st.session_state.last_update = datetime.now()
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ ETL pipeline failed")
        
        st.markdown("---")
        st.subheader("📅 Update Status")
        if st.session_state.last_update:
            st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No manual updates yet")
        
        st.markdown("---")
        st.subheader("🗓️ Date Range Filter")
        date_range_option = st.selectbox("Quick Select:", ["All Data", "1 Month", "3 Months", "6 Months", "1 Year"], key="date_selector")
    
    # Load data
    data_path = f"data/{selected_stock}_processed.csv"
    if not os.path.exists(data_path) and selected_stock == "CIB":
        data_path = "data/processed_stock.csv"
    df = load_processed_data(data_path=data_path)
    
    if df is None:
        st.error("❌ Data not available for this stock")
        st.info("Actions:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Initialize data"):
                with st.spinner("Running initial ETL pipeline..."):
                    if run_etl_pipeline(selected_stock):
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to run ETL pipeline")
        with col2:
            st.info(f"Run: `python -m src.etl` to generate all stocks")
        return

    st.subheader(f"Selected Stock: {selected_stock} ({selected_ticker})")
    
    # Apply date range filter
    with st.sidebar:
        if date_range_option == "1 Month":
            start_date = df.index[-1] - timedelta(days=30)
        elif date_range_option == "3 Months":
            start_date = df.index[-1] - timedelta(days=90)
        elif date_range_option == "6 Months":
            start_date = df.index[-1] - timedelta(days=180)
        elif date_range_option == "1 Year":
            start_date = df.index[-1] - timedelta(days=365)
        else:
            start_date = df.index[0]
        
        col1, col2 = st.columns(2)
        with col1:
            custom_start = st.date_input("Start Date", value=start_date, min_value=df.index[0], max_value=df.index[-1])
        with col2:
            custom_end = st.date_input("End Date", value=df.index[-1], min_value=df.index[0], max_value=df.index[-1])
        
        start_date = pd.Timestamp(custom_start)
        end_date = pd.Timestamp(custom_end)
        st.markdown("---")
        st.info(f"📅 Showing: {start_date.date()} to {end_date.date()}")
    
    # Filter and render
    df_filtered = df[(df.index >= start_date) & (df.index <= end_date)].copy()
    
    if df_filtered.empty:
        st.error("❌ No data available for the selected date range.")
        return
    
    render_dashboard(df_filtered, selected_stock)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.auto_refresh:
            st.success(f"✓ Auto-Refresh: ON ({st.session_state.refresh_interval}s)")
        else:
            st.warning("Auto-Refresh: OFF")
    with col2:
        st.info(f"Data: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    with col3:
        st.caption(f"Last page load: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh mechanism
    if st.session_state.auto_refresh:
        placeholder = st.empty()
        
        for i in range(st.session_state.refresh_interval, 0, -1):
            placeholder.info(f"⏳ Next refresh in {i} seconds... (Auto-Refresh enabled)")
            time.sleep(1)
        
        with st.spinner("🔄 Auto-refreshing data..."):
            if run_etl_pipeline(selected_stock):
                st.session_state.last_update = datetime.now()
                st.cache_data.clear()
                placeholder.success("✓ Data refreshed!")
                time.sleep(1)
                st.rerun()
            else:
                placeholder.warning("⚠️ Auto-refresh skipped (ETL failed)")
                time.sleep(5)
                st.rerun()


if __name__ == "__main__":
    main()

# EGX Stock Analysis - CIB

A complete stock analysis and prediction system for COMI.CA (Canadian Imperial Bank of Commerce) using machine learning and technical indicators.

## 🎯 Project Overview

- **Data Source**: Yahoo Finance (yfinance)
- **Ticker**: COMI.CA (5 years of historical data)
- **Analysis Type**: Classification (predict stock direction UP/DOWN)
- **Framework**: Streamlit web interface
- **ML Model**: RandomForestClassifier (100 trees, max_depth=15)

## 📊 Features

### Technical Indicators (20 total)
- **Returns**: pct_change, log_return
- **Moving Averages**: MA7, MA14, MA30, EMA12, EMA26
- **Momentum**: RSI14, MACD, MACD_Signal
- **Volatility**: rolling_std14, hl_spread
- **Volume**: volume_change, volume_ma
- **Target**: price_up (classification)

### Dashboard Features
- Interactive Plotly charts
- Real-time price data
- RSI indicator with overbought/oversold levels
- MACD indicator with signal line
- Volume analysis with moving average
- Date range filtering
- Auto-refresh capability (5-60 seconds)

## 🚀 Quick Start

### 1. First Time Setup

```bash
# Navigate to project directory
cd /home/adham/term6/egy\ stock\ market

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Processed Data

```bash
# Fetch data, engineer features, and save processed dataset
source .venv/bin/activate
python -m src.etl
```

Output:
- Creates: `data/processed_stock.csv`
- Size: ~400 KB
- Records: 1,168
- Columns: 20

### 3. Run Streamlit Dashboard

```bash
# Start the web interface
source .venv/bin/activate
streamlit run app.py
```

Opens at: `http://localhost:8501`

## 📋 Project Structure

```
egy stock market/
├── app.py                 # Main Streamlit dashboard
├── requirements.txt       # Python dependencies
├── data/
│   └── processed_stock.csv # Processed stock data (auto-generated)
├── src/
│   ├── __init__.py       # Package initialization
│   ├── etl.py            # Data fetching & pipeline
│   ├── features.py       # Feature engineering
│   ├── model.py          # ML model training
│   └── utils.py          # Utility functions
└── .venv/                # Virtual environment
```

## 🔧 Detailed Commands

### ETL Pipeline

```bash
source .venv/bin/activate

# Run complete pipeline (fetch -> feature engineering -> save)
python -m src.etl

# Or individual components
python -c "from src.etl import fetch_data; df = fetch_data('COMI.CA'); print(df.shape)"
python -c "from src.features import add_features; # engineer features"
python -c "from src.model import StockPricePredictor; # train model"
```

### Model Training

```bash
source .venv/bin/activate

python -c "
from src.etl import fetch_data
from src.features import add_features
from src.model import StockPricePredictor

# Load and prepare data
df = fetch_data('COMI.CA')
df = add_features(df)
X = df.drop('price_up', axis=1)
y = df['price_up']

# Train model
model = StockPricePredictor()
model.train(X, y, test_size=0.2)

# Save model
import os
os.makedirs('models', exist_ok=True)
model.save_model('models/stock_predictor.pkl')

print('✓ Model trained and saved')
"
```

### Streamlit App

```bash
source .venv/bin/activate

# Standard run
streamlit run app.py

# With specific port
streamlit run app.py --server.port 8502

# Headless mode (no browser)
streamlit run app.py --logger.level=error --server.headless true
```

### Kafka News Streaming (Simulation)

This project includes:
- `src/kafka_producer.py` (fetches news and sends to Kafka)
- `src/kafka_consumer.py` (reads messages and applies sentiment analysis)

#### 1) Start Kafka locally (Docker)

```bash
docker network create kafka-net

docker run -d --name zookeeper --network kafka-net -p 2181:2181 \
  -e ZOOKEEPER_CLIENT_PORT=2181 \
  confluentinc/cp-zookeeper:7.5.0

docker run -d --name kafka --network kafka-net -p 9092:9092 \
  -e KAFKA_BROKER_ID=1 \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  confluentinc/cp-kafka:7.5.0
```

Create the topic:

```bash
docker exec kafka kafka-topics --create \
  --topic news_topic \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1
```

#### 2) Run consumer (Terminal 1)

```bash
source .venv/bin/activate
python -m src.kafka_consumer --topic news_topic --bootstrap-servers localhost:9092
```

#### 3) Run producer (Terminal 2)

```bash
source .venv/bin/activate
python -m src.kafka_producer --topic news_topic --bootstrap-servers localhost:9092 --interval 30
```

Stop after one cycle (quick test):

```bash
python -m src.kafka_producer --topic news_topic --bootstrap-servers localhost:9092 --iterations 1
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| yfinance | ≥0.2.32 | Data fetching |
| pandas | ≥2.1.3 | Data manipulation |
| numpy | ≥1.26.0 | Numerical computing |
| scikit-learn | ≥1.3.2 | ML models |
| streamlit | ≥1.28.1 | Web interface |
| plotly | ≥5.18.0 | Interactive charts |
| matplotlib | ≥3.8.2 | Static charts |
| joblib | ≥1.3.2 | Model serialization |

## 🐛 Troubleshooting

### Issue: "streamlit command not found"
```bash
# Solution: Always activate venv first
source .venv/bin/activate
streamlit run app.py
```

### Issue: "Module not found: src.etl"
```bash
# Solution: Run from project root
cd /home/adham/term6/egy\ stock\ market
source .venv/bin/activate
python -m src.etl  # NOT python src/etl.py
```

### Issue: "No data for selected date range"
```bash
# Re-run ETL pipeline
source .venv/bin/activate
python -m src.etl
```

### Issue: "Input X contains infinity" (old issue - FIXED)
```bash
# Already fixed in features.py - replaces inf with 0
# and handles division by zero
```

## 📈 Model Performance

Current test metrics:
- **Train Accuracy**: ~98.6% (may indicate overfitting)
- **Test Accuracy**: ~47.4%
- **Precision**: ~47.4%
- **Recall**: ~46.2%
- **F1-Score**: ~46.8%

Note: Model accuracy is around 47%, which is close to random (50% baseline) for a 50-50 binary target. Stock price prediction is inherently difficult; consider:
- Collecting more features
- Using different algorithms
- Hyperparameter tuning
- Ensemble methods

## 🔄 Real-Time Updates

The dashboard supports auto-refresh:

1. Enable "Auto-Refresh" in sidebar
2. Select refresh interval (5-60 seconds)
3. Dashboard automatically:
   - Fetches new data
   - Re-engineers features
   - Updates all charts
   - Refreshes metrics

## 📝 Data Processing Pipeline

```
1. FETCH (fetch_data)
   ├─ Download 5 years of COMI.CA data
   ├─ Select OHLCV columns
   ├─ Handle missing values
   └─ Output: 1208 records

2. ENGINEER FEATURES (add_features)
   ├─ Returns (pct_change, log_return)
   ├─ Moving Averages (MA7, MA14, MA30, EMA12, EMA26)
   ├─ Momentum (RSI14, MACD, MACD_Signal)
   ├─ Volatility (rolling_std14, hl_spread)
   ├─ Volume (volume_change, volume_ma)
   ├─ Target (price_up classification)
   ├─ Replace inf with 0
   ├─ Drop NaN rows
   └─ Output: 1168 records × 20 columns

3. SAVE (save_stock_data)
   └─ CSV: data/processed_stock.csv

4. VISUALIZE (Streamlit)
   ├─ Load processed data
   ├─ Render 5 dashboard sections
   ├─ Interactive date filtering
   └─ Auto-refresh capability
```

## 🔐 Error Handling

All modules include:
- ✅ Try-except blocks
- ✅ Logging with timestamps
- ✅ Graceful failure handling
- ✅ Input validation
- ✅ Data quality checks

## 📞 Support

For issues, check:
1. Virtual environment is activated: `source .venv/bin/activate`
2. All dependencies installed: `pip install -r requirements.txt`
3. Project directory: `/home/adham/term6/egy stock market`
4. Data exists: `ls -l data/processed_stock.csv`

## 📄 License

Educational project - Feel free to use and modify.

---

**Last Updated**: April 24, 2026  
**System**: Ubuntu/Linux (Python 3.12.3)  
**Status**: ✅ Fully Functional

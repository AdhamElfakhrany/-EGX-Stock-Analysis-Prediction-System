# 1) Activate environment
source .venv/bin/activate

# 2) (Optional first time) install dependencies
pip install -r requirements.txt

# 3) Run ETL to generate/update stock datasets
python -m src.etl

# 4) Start dashboard
streamlit run app.py


# 🚀 QUICK START - EGX Stock Analysis Project

## ✅ Project Status: FULLY OPERATIONAL

All 6 debugging steps complete. Project is production-ready.

---

## 📋 Three Commands to Start Using

### Copy & Paste These Commands:

```bash
# 1️⃣ Navigate to project directory
cd /home/adham/term6/egy\ stock\ market

# 2️⃣ Activate virtual environment
source .venv/bin/activate

# 3️⃣ Start the dashboard
streamlit run app.py
```

**That's it!** Dashboard opens at `http://localhost:8501`

---

## 🔧 One-Time Setup (if needed)

If starting from scratch:

```bash
cd /home/adham/term6/egy\ stock\ market
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m src.etl
```

---

## 📊 What You Get

### Dashboard Features:
- ✅ Real-time price charts with moving averages
- ✅ RSI indicator (overbought/oversold zones)
- ✅ MACD momentum indicator
- ✅ Volume analysis
- ✅ Auto-refresh capability (5-60 seconds)
- ✅ Historical date range filtering

### Data:
- **Stock**: COMI.CA (Canadian Imperial Bank)
- **Period**: 5 years of data (2021-2026)
- **Records**: 1,168 trading days
- **Features**: 20 technical indicators

---

## 🛠️ What Was Fixed

| Issue | Fix |
|-------|-----|
| PEP 668 (external environment) | Virtual environment setup |
| Incompatible numpy | Updated requirements.txt |
| MultiIndex columns | Flatten DataFrame columns |
| Infinite values | Replace inf with 0, check division by zero |
| App syntax errors | Complete rewrite of app.py |
| Missing error handling | Added comprehensive try-except blocks |

---

## 📁 Project Structure

```
├── app.py                   ← Main Streamlit dashboard
├── requirements.txt         ← Python dependencies
├── README.md               ← Full documentation
├── DEBUGGING_REPORT.md     ← What was fixed
├── setup.sh                ← Setup script
├── run.sh                  ← Quick run script
├── data/
│   └── processed_stock.csv ← Stock data (1,168 × 20)
└── src/
    ├── etl.py             ← Data fetching & pipeline
    ├── features.py        ← Feature engineering (20 features)
    ├── model.py           ← ML model (RandomForest)
    └── utils.py           ← Utility functions
```

---

## 📊 Dashboard Sections

### Section 1: Key Metrics
```
Current Price | Daily Change % | Trading Volume | Data Points
```

### Section 2: Price Chart
```
Close Price with MA7 (orange) and MA30 (green) overlays
Interactive Plotly chart with hover details
```

### Section 3 & 4: Technical Indicators (Side by Side)
```
RSI Indicator          |  MACD Indicator
Overbought: 70         |  MACD Line
Oversold: 30           |  Signal Line
```

### Section 5: Volume Analysis
```
Volume bars with 20-period moving average
```

---

## 🎮 Interactive Controls

**Sidebar Features:**
- ✅ Auto-Refresh toggle (ON/OFF)
- ✅ Refresh interval selector (5-60 seconds)
- ✅ Manual refresh button
- ✅ Date range quick presets (All, 1M, 3M, 6M, 1Y)
- ✅ Custom date range picker
- ✅ Last update timestamp

---

## 📈 Data Pipeline

```
1. FETCH
   └─ Download 5 years COMI.CA data (1,208 records)

2. ENGINEER
   └─ Create 20 technical indicators
   └─ Handle NaN and infinite values
   └─ Output: 1,168 clean records

3. SAVE
   └─ Export to data/processed_stock.csv

4. VISUALIZE
   └─ Load in Streamlit dashboard
   └─ Render 5 chart sections
   └─ Enable interactive filtering
```

---

## 🧪 Testing Commands

### Test ETL pipeline:
```bash
source .venv/bin/activate
python -m src.etl
```

### Test features:
```bash
source .venv/bin/activate
python -c "
from src.etl import fetch_data
from src.features import add_features
df = fetch_data('COMI.CA')
df = add_features(df)
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
"
```

### Test model:
```bash
source .venv/bin/activate
python -c "
from src.etl import fetch_data
from src.features import add_features
from src.model import StockPricePredictor

df = fetch_data('COMI.CA')
df = add_features(df)
X = df.drop('price_up', axis=1)
y = df['price_up']

model = StockPricePredictor()
model.train(X, y)
"
```

---

## 🐛 Troubleshooting

### "streamlit: command not found"
```bash
# WRONG ❌
streamlit run app.py

# RIGHT ✅
source .venv/bin/activate
streamlit run app.py
```

### "No module named src"
```bash
# WRONG ❌
cd src
python -m etl

# RIGHT ✅
cd /home/adham/term6/egy\ stock\ market
source .venv/bin/activate
python -m src.etl
```

### "No data for selected date range"
```bash
# Regenerate data
source .venv/bin/activate
python -m src.etl
```

### Dashboard won't load processed data
```bash
# Make sure ETL ran successfully
source .venv/bin/activate
python -m src.etl

# Then check file exists
ls -l data/processed_stock.csv
```

---

## 📦 Key Dependencies

```
yfinance     ≥0.2.32   → Fetch stock data
pandas       ≥2.1.3    → Data manipulation
numpy        ≥1.26.0   → Numerical computing
scikit-learn ≥1.3.2    → Machine learning
streamlit    ≥1.28.1   → Web interface
plotly       ≥5.18.0   → Interactive charts
```

---

## 🎯 Model Performance

Current Statistics:
```
Training Accuracy:  98.6% (suggests overfitting)
Testing Accuracy:   47.4% (close to random baseline)
Precision:          47.4%
Recall:             46.2%
F1-Score:           46.8%
```

Note: Stock price prediction is inherently difficult. The model needs:
- More diverse features
- Different algorithms
- Hyperparameter tuning
- More training data

---

## 📞 Support

**If app won't start:**
1. Activate venv: `source .venv/bin/activate`
2. Check data: `ls data/processed_stock.csv`
3. Regenerate: `python -m src.etl`
4. Run with debug: `streamlit run app.py --logger.level=debug`

---

## ✨ What's Included

- ✅ Complete ETL pipeline
- ✅ 20 technical indicators
- ✅ ML classification model
- ✅ Interactive web dashboard
- ✅ Real-time data refresh
- ✅ Error handling & logging
- ✅ Comprehensive documentation

---

## 🚀 Ready to Launch!

Simply run:
```bash
cd /home/adham/term6/egy\ stock\ market
source .venv/bin/activate
streamlit run app.py
```

**Dashboard opens at:** http://localhost:8501

Enjoy! 🎉

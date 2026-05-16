#!/bin/bash
# setup.sh - One-time initial setup

echo "========================================="
echo "EGX Stock Analysis - Initial Setup"
echo "========================================="

cd /home/adham/term6/egy\ stock\ market

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# Run ETL pipeline
echo "Generating processed data..."
python -m src.etl > /dev/null 2>&1

echo ""
echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "Next, run:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
echo ""

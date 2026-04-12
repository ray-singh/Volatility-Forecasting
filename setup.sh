#!/bin/bash

# Setup script for Volatility Forecasting Dashboard
# Run: bash setup.sh

set -e  # Exit on error

echo "📊 Volatility Forecasting Setup"
echo "================================"

# Check Python version
echo -n "✓ Checking Python... "
python_version=$(python -c 'import sys; v = sys.version_info; print(int(str(v.major) + str(v.minor).zfill(2)))')
min_version=309  # 3.9 = 309
if [[ $python_version -ge $min_version ]]; then
    python_display=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Python $python_display ✓"
else
    python_display=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "FAILED - Need Python >= 3.9, found $python_display"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "• Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "• Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "• Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install dependencies
echo "• Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

# Create directories
echo "• Creating directories..."
mkdir -p data/raw
mkdir -p models

# Verify installation
echo -n "✓ Verifying imports... "
python -c "import polars, numpy, lightgbm, torch, streamlit, plotly; print('OK')" || {
    echo "FAILED"
    exit 1
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "📚 Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start dashboard:"
echo "   streamlit run app.py"
echo ""
echo "3. Open browser to:"
echo "   http://localhost:8501"
echo ""
echo "For quick start guide, see: QUICKSTART.md"
echo "For dashboard reference, see: DASHBOARD.md"

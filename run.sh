#!/usr/bin/env bash
set -e

echo "===================================================================="
echo " RailGuard Vision - Indian Railways Security Surveillance Prototype"
echo " FOR HUMAN REVIEW ONLY - Does not detect explosives or narcotics"
echo "===================================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# Virtual environment setup
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating Python virtual environment 'venv'..."
    python3 -m venv venv
    echo "[SETUP] Virtual environment created successfully."
fi

# Activate virtual environment
# shellcheck source=/dev/null
source venv/bin/activate

# Install dependencies
echo "[SETUP] Verifying dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

# Ensure data directories exist
mkdir -p data/alerts data/clips data/ground_truth

echo "===================================================================="
echo "[LAUNCH] Starting RailGuard Vision Streamlit Dashboard..."
echo "Local URL: http://localhost:8501"
echo "===================================================================="

streamlit run app/dashboard.py

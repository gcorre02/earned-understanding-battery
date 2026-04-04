#!/bin/bash
# Setup script for M5 Max calibration run
# Usage: cd earned-understanding-battery && bash scripts/setup_m5.sh

set -e

echo "=== Earned Understanding Battery — M5 Max Setup ==="

# Check Python version
python3 --version || { echo "Python 3.11+ required"; exit 1; }

# Check we're on Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "WARNING: Expected Apple Silicon (arm64), got $(uname -m)"
    echo "Continuing anyway..."
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Installing dependencies from requirements-m5.txt..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-m5.txt

# Install the package in editable mode
.venv/bin/pip install -e .

# Verify torch
echo ""
echo "Verifying torch installation..."
.venv/bin/python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else \"N/A\"}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Quick smoke test
echo ""
echo "Running quick smoke test..."
.venv/bin/python -m pytest tests/test_core.py tests/test_domains.py -v --tb=short 2>&1 | tail -5

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run calibration with:"
echo "  .venv/bin/python scripts/run_m5_calibration.py"
echo ""
echo "Expected runtime: ~2 hours (2A TinyLlama ~90 min, 3C Foxworthy F ~30 min)"
echo "Results will be in: results/calibration/"

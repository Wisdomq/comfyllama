#!/bin/bash

# Server Setup Script
# Prepares the server environment for training

set -e

echo "======================================================================"
echo "SERVER SETUP - COMFYUI TRAINING"
echo "======================================================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found!"
    exit 1
fi

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "[4/5] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[5/5] Installing dependencies..."
echo "This will take 5-10 minutes..."
echo ""

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.38.0
pip install datasets==2.18.0
pip install accelerate==0.27.0
pip install peft==0.9.0
pip install trl==0.7.11
pip install sentencepiece protobuf scipy scikit-learn tqdm pillow

echo ""
echo "======================================================================"
echo "✓ SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "Verifying CPU optimization..."
python cpu_setup_server.py

echo ""
echo "======================================================================"
echo "NEXT STEPS"
echo "======================================================================"
echo ""
echo "1. Configure email notification:"
echo "   export TRAINING_NOTIFICATION_EMAIL='your@email.com'"
echo ""
echo "2. Run the full training pipeline:"
echo "   bash run_full_training_pipeline.sh"
echo ""
echo "   Or run steps individually:"
echo "   python merge_datasets.py"
echo "   python format_combined_dataset.py"
echo "   python train_model_server.py"
echo ""
echo "3. Monitor training:"
echo "   tail -f training.log"
echo ""
echo "======================================================================"

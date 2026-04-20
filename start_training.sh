#!/bin/bash

# Safe training launcher with proper logging and error handling
# This script ensures all output is captured and training runs properly

echo "======================================================================"
echo "COMFYUI TRAINING - SAFE LAUNCHER"
echo "======================================================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Check if datasets exist
if [ ! -f "comfyui_dataset_combined_formatted.jsonl" ]; then
    echo "❌ Formatted dataset not found!"
    echo ""
    echo "Run dataset preparation first:"
    echo "  python merge_datasets.py"
    echo "  python format_combined_dataset.py"
    echo ""
    exit 1
fi

# Create log directory
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/training_${TIMESTAMP}.log"

echo "Training will start with full logging enabled"
echo "Log file: $LOGFILE"
echo ""
echo "Options:"
echo "  1. Run in foreground (see output, can't disconnect)"
echo "  2. Run in background with nohup (can disconnect)"
echo ""
read -p "Choose option (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "Starting training in foreground..."
        echo "Press Ctrl+C to stop"
        echo "======================================================================"
        echo ""
        python train_model_server.py 2>&1 | tee "$LOGFILE"
        ;;
    2)
        echo ""
        echo "Starting training in background..."
        nohup python train_model_server.py > "$LOGFILE" 2>&1 &
        PID=$!
        echo "✓ Training started with PID: $PID"
        echo ""
        echo "Monitor progress:"
        echo "  tail -f $LOGFILE"
        echo ""
        echo "Check if running:"
        echo "  ps aux | grep $PID"
        echo ""
        echo "Stop training:"
        echo "  kill $PID"
        echo ""
        echo "You can now safely disconnect from the server."
        echo "======================================================================"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

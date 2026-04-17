#!/bin/bash

# Full Training Pipeline for Server
# Runs dataset preparation, formatting, and training in sequence

set -e  # Exit on error

echo "======================================================================"
echo "COMFYUI TRAINING PIPELINE - SERVER"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Merge datasets (synthetic + HuggingFace)"
echo "  2. Format dataset for training"
echo "  3. Train the model"
echo "  4. Send email notification"
echo ""
echo "Estimated total time: 4-5 hours"
echo "======================================================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment not activated!"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Check if email is configured
if [ -z "$TRAINING_NOTIFICATION_EMAIL" ]; then
    echo "⚠️  Note: TRAINING_NOTIFICATION_EMAIL not set"
    echo "Training will complete successfully, but no email will be sent."
    echo "You can monitor progress with: tail -f training.log"
    echo ""
fi

echo ""
echo "======================================================================"
echo "STEP 1/3: MERGING DATASETS"
echo "======================================================================"
echo "Downloading and merging 500 real workflows with 500 synthetic examples"
echo "This will download ~6.25 GB of data..."
echo ""

python merge_datasets.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset merge failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 2/3: FORMATTING DATASET"
echo "======================================================================"
echo "Applying TinyLlama chat template to combined dataset..."
echo ""

python format_combined_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Dataset formatting failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 3/3: TRAINING MODEL"
echo "======================================================================"
echo "Starting training on AMD EPYC 7402P (48 cores)"
echo "This will take approximately 3-5 hours..."
echo ""
echo "You can safely disconnect. Training will continue in background."
echo "Monitor with: tail -f training.log"
echo ""

# Start training
python train_model_server.py

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Check your email for completion notification"
echo "  2. Test the model: python test_model.py"
echo "  3. Download model if needed"
echo ""
echo "Model location: ./comfyui_lora_model_final"
echo "======================================================================"

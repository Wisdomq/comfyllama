# Server Training Guide - AMD EPYC 7402P

Complete guide for training ComfyUI workflow generator on your 48-core server.

## Server Specifications

- **CPU**: AMD EPYC 7402P 24-Core Processor
- **Threads**: 48 (24 cores × 2 threads/core)
- **Architecture**: x86_64
- **Max Frequency**: 2.8 GHz

## Setup Instructions

### 1. Clone Repository on Server

```bash
git clone <your-repo-url>
cd fine_tuning
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.38.0
pip install datasets==2.18.0
pip install accelerate==0.27.0
pip install peft==0.9.0
pip install trl==0.7.11
pip install sentencepiece protobuf scipy scikit-learn tqdm pillow
```

### 4. Verify CPU Optimization

```bash
python cpu_setup_server.py
```

Expected output:
```
CPU threads: 48
✓ CPU optimized for 48 threads
```

## Dataset Preparation

### Step 1: Merge Datasets

This downloads 500 real ComfyUI workflows from HuggingFace and combines with your 500 synthetic examples.

```bash
python merge_datasets.py
```

**Time**: ~10-15 minutes (downloads 6.25 GB)

**Output**: `comfyui_dataset_combined.jsonl` (~1000 examples)

### Step 2: Format Dataset

```bash
python format_combined_dataset.py
```

**Time**: ~2-3 minutes

**Output**: `comfyui_dataset_combined_formatted.jsonl`

**Note**: This script will tell you the recommended `max_seq_length` for training.

## Training

### Configure Email Notifications

Set your email address to receive completion notification:

```bash
export TRAINING_NOTIFICATION_EMAIL="your.email@example.com"
```

Or edit it directly in the training script.

### Start Training

```bash
# Run in background with nohup
nohup python train_model_server.py > training.log 2>&1 &

# Or use screen/tmux for interactive monitoring
screen -S training
python train_model_server.py
# Ctrl+A, D to detach
```

### Monitor Training

```bash
# Watch log file
tail -f training.log

# Check process
ps aux | grep train_model_server

# Monitor CPU usage
htop
```

## Training Configuration

### Optimized Settings for 48 Cores

```python
per_device_train_batch_size=2      # 2x larger than laptop
gradient_accumulation_steps=4       # Effective batch = 8
dataloader_num_workers=8            # 8 parallel data loaders
num_train_epochs=3                  # 3 full passes
learning_rate=2e-4                  # Standard for LoRA
```

### Expected Performance

- **Training speed**: ~1-2 seconds per step
- **Total time**: ~3-5 hours for 1000 examples × 3 epochs
- **Memory usage**: ~8-10 GB RAM
- **CPU usage**: 100% across all 48 threads

### What to Monitor

✅ **Good signs**:
- Loss decreasing from ~2.0 to ~0.5-0.8
- Grad norm between 0.5-2.0 (NOT 0.0!)
- Steady progress without crashes
- CPU at 100% utilization

❌ **Warning signs**:
- Grad norm = 0.0 (no learning)
- Loss increasing or staying flat
- Very slow progress (>5 sec/step)
- Out of memory errors

## After Training

### 1. Check Email

You should receive an email notification with:
- Training duration
- Final loss
- Model location
- Next steps

### 2. Test the Model

```bash
python test_model.py
```

This runs 4 test cases and validates JSON output.

### 3. Download Model (if needed)

```bash
# From server to local machine
scp -r user@server:/path/to/comfyui_lora_model_final ./
```

## Troubleshooting

### Email Not Sending

Check mail service:
```bash
systemctl status postfix
# or
systemctl status sendmail
```

Test mail:
```bash
echo "test" | mail -s "test" your@email.com
```

### Training Very Slow

Check CPU threads:
```bash
python cpu_setup_server.py
```

Should show 48 threads, not 1 or 8.

### Out of Memory

Reduce batch size in `train_model_server.py`:
```python
per_device_train_batch_size=1  # Instead of 2
```

### Grad Norm = 0.0

This means no learning is happening. Restart training from scratch, don't resume from checkpoint.

## File Structure

```
fine_tuning/
├── cpu_setup_server.py              # CPU optimization for 48 cores
├── merge_datasets.py                # Combine datasets
├── format_combined_dataset.py       # Format for training
├── train_model_server.py            # Main training script
├── send_email_notification.py       # Email notification
├── test_model.py                    # Test trained model
├── comfyui_dataset_combined.jsonl   # Merged dataset
├── comfyui_dataset_combined_formatted.jsonl  # Formatted dataset
├── training_output_server/          # Checkpoints
└── comfyui_lora_model_final/        # Final trained model
```

## Expected Timeline

1. **Setup**: 10 minutes
2. **Dataset merge**: 15 minutes
3. **Dataset format**: 3 minutes
4. **Training**: 3-5 hours
5. **Testing**: 5 minutes

**Total**: ~4-5.5 hours

## Performance Comparison

| System | Cores | Training Time (1000 examples × 3 epochs) |
|--------|-------|------------------------------------------|
| Laptop (Ryzen 7 5700U) | 16 | ~15-20 hours |
| Server (EPYC 7402P) | 48 | ~3-5 hours |

**Speedup**: 3-4x faster on server!

## Next Steps After Training

1. ✅ Test model with `test_model.py`
2. ✅ Verify JSON output is valid
3. ✅ Test with custom prompts
4. ✅ Deploy to production
5. ✅ Consider training for more epochs if needed

## Support

If you encounter issues:
1. Check `training.log` for errors
2. Verify CPU optimization with `cpu_setup_server.py`
3. Test email with `send_email_notification.py`
4. Monitor with `htop` or `top`

---

**Ready to train?** Follow the steps above and your model will be ready in ~4-5 hours!

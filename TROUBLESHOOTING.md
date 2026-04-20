# Training Troubleshooting Guide

## Current Situation

Your training started but all processes disappeared and `training_output_server/` is empty. This means training crashed immediately at startup.

## Root Cause Analysis

The most likely causes:
1. **Import error** - Missing dependency or import failure
2. **Dataset missing** - Formatted dataset file not found
3. **Memory issue** - OOM killer terminated the process
4. **Permission issue** - Can't write to output directory

## Step-by-Step Diagnosis

### Step 1: Run Diagnostics

```bash
# Activate environment
source venv/bin/activate

# Run diagnostic script
python diagnose_training.py
```

This will test:
- CPU setup import
- PyTorch and libraries
- Dataset files existence
- Dataset loading
- Model accessibility

**If diagnostics fail**, you'll see exactly which step is broken.

### Step 2: Check System Logs

```bash
# Check if process was killed by OOM
dmesg | tail -100 | grep -i "python\|killed\|oom"

# Check system messages
journalctl -n 100 | grep -i python
```

Look for messages like:
- "Out of memory: Killed process"
- "python invoked oom-killer"

### Step 3: Test Training Manually

```bash
# Run training with full output capture
python train_model_server.py 2>&1 | tee training_test.log
```

This will:
- Show all output in real-time
- Save everything to `training_test.log`
- Reveal the exact error

### Step 4: Check Dataset Files

```bash
# Verify datasets exist
ls -lh comfyui_dataset_combined*.jsonl

# If missing, regenerate:
python merge_datasets.py
python format_combined_dataset.py
```

## Common Issues and Fixes

### Issue 1: Dataset Files Missing

**Symptom**: `FileNotFoundError: comfyui_dataset_combined_formatted.jsonl`

**Fix**:
```bash
# Generate datasets
python merge_datasets.py
python format_combined_dataset.py

# Verify
ls -lh comfyui_dataset_combined_formatted.jsonl
```

### Issue 2: Import Errors

**Symptom**: `ModuleNotFoundError` or `ImportError`

**Fix**:
```bash
# Reinstall dependencies
pip install --upgrade transformers peft trl datasets torch
```

### Issue 3: Out of Memory

**Symptom**: Process killed, no error message, `dmesg` shows OOM

**Fix**: Reduce batch size in `train_model_server.py`:
```python
per_device_train_batch_size=1,  # Reduce from 2 to 1
```

### Issue 4: Multiple Processes Spawned

**Symptom**: 8-9 Python processes, all consuming memory

**Explanation**: This is NORMAL - PyTorch spawns dataloader workers:
- 1 main training process
- 8 dataloader worker processes (configured in script)

**Problem**: If they all disappear, the parent crashed.

## Safe Training Launch

Use the new launcher script:

```bash
# Make executable
chmod +x start_training.sh

# Run with proper logging
./start_training.sh
```

This script:
- Checks prerequisites
- Offers foreground or background mode
- Captures all output to timestamped log
- Provides monitoring commands

## Manual Background Training

If you prefer manual control:

```bash
# Option 1: nohup (simple)
nohup python train_model_server.py > training.log 2>&1 &
echo $! > training.pid

# Monitor
tail -f training.log

# Stop
kill $(cat training.pid)

# Option 2: screen (advanced)
screen -S training
python train_model_server.py
# Press Ctrl+A then D to detach
# Reattach: screen -r training

# Option 3: tmux (advanced)
tmux new -s training
python train_model_server.py
# Press Ctrl+B then D to detach
# Reattach: tmux attach -t training
```

## Monitoring Training

### Check if training is running:
```bash
ps aux | grep train_model_server
```

### Monitor progress:
```bash
tail -f training.log
```

### Check GPU/CPU usage:
```bash
htop
```

### Check memory usage:
```bash
free -h
```

### Check training output:
```bash
ls -lh training_output_server/
```

Should see:
- `checkpoint-100/`
- `checkpoint-200/`
- etc.

## Expected Training Behavior

### Startup (first 2-3 minutes):
```
[1/7] Loading TinyLlama model...
✓ Model loaded
  Parameters: 1,100,048,384
  Memory: 4.40 GB

[2/7] Applying LoRA configuration...
✓ LoRA applied
trainable params: 1,114,112 || all params: 1,101,162,496 || trainable%: 0.1012

[3/7] Loading formatted dataset...
✓ Loaded 1000 examples

[4/7] Configuring training...
✓ Training configuration:
  Epochs: 3
  Batch size: 2
  Total steps: ~375

[5/7] Creating SFTTrainer...
✓ Trainer created

[6/7] Starting training...
```

### During Training:
```
{'loss': 2.1234, 'grad_norm': 1.2345, 'learning_rate': 0.0002, 'epoch': 0.03}
{'loss': 1.8765, 'grad_norm': 0.9876, 'learning_rate': 0.00019, 'epoch': 0.05}
...
```

**CRITICAL**: `grad_norm` must be > 0.0 (typically 0.5-2.0)

If `grad_norm` is 0.0, training is broken!

### Completion:
```
✓ TRAINING COMPLETED SUCCESSFULLY!
Training time: 3.45 hours

[7/7] Saving fine-tuned model...
✓ Model saved to: ./comfyui_lora_model_final
```

## Next Steps After Diagnosis

1. **Run diagnostics**: `python diagnose_training.py`
2. **Fix any issues** identified
3. **Test training**: `python train_model_server.py 2>&1 | tee test.log`
4. **If successful**, use proper launcher: `./start_training.sh`
5. **Monitor progress**: `tail -f logs/training_*.log`

## Getting Help

If issues persist, provide:
1. Output of `diagnose_training.py`
2. Contents of `training_test.log` (first 100 lines)
3. Output of `dmesg | tail -100`
4. Server specs: `free -h` and `nproc`

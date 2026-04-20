# Quick Command Reference

## Immediate Actions (Run These Now)

### 1. Activate Environment
```bash
cd ~/code/fine_tuning/tinyllama/comfyllama
source venv/bin/activate
```

### 2. Run Diagnostics
```bash
python diagnose_training.py
```

This will tell you exactly what's wrong.

### 3. Check What Happened
```bash
# Check if process was killed
dmesg | tail -50 | grep -i "python\|killed\|oom"

# Check system logs
journalctl -n 50 | grep -i python
```

## If Diagnostics Pass

### Option A: Test Training (See Output)
```bash
python train_model_server.py 2>&1 | tee training_test.log
```

Watch for errors. Press Ctrl+C to stop if needed.

### Option B: Use Safe Launcher
```bash
chmod +x start_training.sh
./start_training.sh
```

Choose option 2 for background mode.

## If Dataset Missing

```bash
# Generate synthetic dataset (if needed)
python create_comfyui_dataset.py

# Merge with HuggingFace data
python merge_datasets.py

# Format for training
python format_combined_dataset.py

# Verify
ls -lh comfyui_dataset_combined_formatted.jsonl
```

## Monitoring Commands

### Check if training is running:
```bash
ps aux | grep train_model_server
```

### Monitor progress:
```bash
tail -f training.log
# or
tail -f logs/training_*.log
```

### Check output directory:
```bash
ls -lh training_output_server/
```

### Check system resources:
```bash
htop  # Press 'q' to quit
free -h
```

## Background Training (Manual)

### Start:
```bash
nohup python train_model_server.py > training.log 2>&1 &
echo $! > training.pid
```

### Monitor:
```bash
tail -f training.log
```

### Check status:
```bash
ps aux | grep $(cat training.pid)
```

### Stop:
```bash
kill $(cat training.pid)
```

## Expected Timeline

1. **Diagnostics**: 1-2 minutes
2. **Dataset prep** (if needed): 10-15 minutes
3. **Training startup**: 2-3 minutes
4. **Training**: 3-5 hours
5. **Model saving**: 1-2 minutes

## What Success Looks Like

### During Training:
```
{'loss': 2.1234, 'grad_norm': 1.2345, 'learning_rate': 0.0002, 'epoch': 0.03}
{'loss': 1.8765, 'grad_norm': 0.9876, 'learning_rate': 0.00019, 'epoch': 0.05}
```

**Key**: `grad_norm` > 0.0 (typically 0.5-2.0)

### After Completion:
```bash
ls -lh comfyui_lora_model_final/
```

Should show:
- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer files`

## Troubleshooting

See `TROUBLESHOOTING.md` for detailed guide.

Quick checks:
1. Virtual environment activated? `echo $VIRTUAL_ENV`
2. Dataset exists? `ls -lh comfyui_dataset_combined_formatted.jsonl`
3. Enough disk space? `df -h .`
4. Enough memory? `free -h`

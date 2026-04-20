# Training Recovery Guide

## What Happened

Your training started but immediately crashed. All 9 processes disappeared and `training_output_server/` is empty.

## Why This Happened

The pipeline script (`run_full_training_pipeline.sh`) ran without output redirection, so when training crashed, no error logs were saved. The multiple processes you saw were normal (1 main + 8 dataloader workers), but they all terminated when the parent process crashed.

## Most Likely Causes

1. **Dataset missing** - The formatted dataset file wasn't found
2. **Import error** - A library failed to import
3. **Memory issue** - OOM killer terminated the process
4. **Startup error** - Training crashed before creating output directory

## Recovery Steps

### Step 1: Diagnose (2 minutes)

```bash
cd ~/code/fine_tuning/tinyllama/comfyllama
source venv/bin/activate
python diagnose_training.py
```

This will test everything and show you exactly what's broken.

### Step 2: Fix Issues

**If dataset missing:**
```bash
python create_comfyui_dataset.py  # You already did this
python merge_datasets.py
python format_combined_dataset.py
```

**If import errors:**
```bash
pip install --upgrade transformers peft trl datasets torch
```

**If memory issues:**
Edit `train_model_server.py`, change line 95:
```python
per_device_train_batch_size=1,  # Reduce from 2
```

### Step 3: Test Training

```bash
python train_model_server.py 2>&1 | tee training_test.log
```

Watch the output. If it starts successfully, you'll see:
```
[1/7] Loading TinyLlama model...
✓ Model loaded
[2/7] Applying LoRA configuration...
✓ LoRA applied
[3/7] Loading formatted dataset...
✓ Loaded 1000 examples
```

Press Ctrl+C after a few minutes if it's working.

### Step 4: Start Proper Training

Once the test works, use the safe launcher:

```bash
chmod +x start_training.sh
./start_training.sh
```

Choose option 2 for background mode.

## New Tools Created

### 1. `diagnose_training.py`
Tests all prerequisites before training starts. Run this first!

### 2. `start_training.sh`
Safe training launcher with:
- Prerequisite checks
- Proper logging to timestamped files
- Background mode option
- Monitoring commands

### 3. `TROUBLESHOOTING.md`
Comprehensive troubleshooting guide with:
- Common issues and fixes
- Monitoring commands
- Expected behavior
- Manual background training

### 4. `QUICK_COMMANDS.md`
Quick reference for all commands you need.

## Monitoring Training

### Check if running:
```bash
ps aux | grep train_model_server
```

### Monitor progress:
```bash
tail -f logs/training_*.log
```

### Check output:
```bash
ls -lh training_output_server/
```

Should see checkpoints appearing:
- `checkpoint-100/`
- `checkpoint-200/`
- etc.

## Expected Training Timeline

1. **Model loading**: 1-2 minutes
2. **Dataset loading**: 30 seconds
3. **Training setup**: 30 seconds
4. **Training**: 3-5 hours (375 steps × 30-50 sec/step)
5. **Model saving**: 1-2 minutes

Total: ~3-5 hours

## What Success Looks Like

### During Training:
```
{'loss': 2.1234, 'grad_norm': 1.2345, 'learning_rate': 0.0002, 'epoch': 0.03}
{'loss': 1.8765, 'grad_norm': 0.9876, 'learning_rate': 0.00019, 'epoch': 0.05}
{'loss': 1.5432, 'grad_norm': 0.7654, 'learning_rate': 0.00018, 'epoch': 0.08}
```

**Critical**: `grad_norm` must be > 0.0 (typically 0.5-2.0)

### After Completion:
```bash
$ ls -lh comfyui_lora_model_final/
total 4.5M
-rw-r--r-- 1 wisdom atf  512 Apr 17 22:30 adapter_config.json
-rw-r--r-- 1 wisdom atf 4.3M Apr 17 22:30 adapter_model.safetensors
-rw-r--r-- 1 wisdom atf  48K Apr 17 22:30 special_tokens_map.json
-rw-r--r-- 1 wisdom atf 1.8M Apr 17 22:30 tokenizer.model
-rw-r--r-- 1 wisdom atf  48K Apr 17 22:30 tokenizer_config.json
```

## Next Steps After Training

1. **Test the model**: `python test_model.py`
2. **Download from server** (if needed)
3. **Integrate into your application**

## Quick Reference

See `QUICK_COMMANDS.md` for all commands.

## Need Help?

If issues persist after diagnostics, provide:
1. Output of `diagnose_training.py`
2. First 100 lines of `training_test.log`
3. Output of `dmesg | tail -100`
4. Server memory: `free -h`

---

**TL;DR**: Run `python diagnose_training.py` first, then follow the steps it suggests.

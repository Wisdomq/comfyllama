# Deployment Checklist

Use this checklist to ensure everything is ready before pushing to GitHub and deploying to server.

## ✅ Pre-Deployment (On Your PC)

### Files to Commit
- [ ] `merge_datasets.py` - Dataset merger
- [ ] `format_combined_dataset.py` - Dataset formatter
- [ ] `train_model_server.py` - Server training script
- [ ] `cpu_setup_server.py` - Server CPU optimization
- [ ] `send_email_notification.py` - Email notification
- [ ] `test_model.py` - Model testing
- [ ] `setup_server.sh` - Server setup script
- [ ] `run_full_training_pipeline.sh` - Full pipeline script
- [ ] `SERVER_TRAINING_GUIDE.md` - Documentation
- [ ] `COMPLETE_SETUP_SUMMARY.md` - Summary
- [ ] `DEPLOYMENT_CHECKLIST.md` - This file

### Files to Keep Local (Don't Commit)
- [ ] `venv/` - Virtual environment (add to .gitignore)
- [ ] `*.jsonl` - Dataset files (too large, regenerate on server)
- [ ] `training_output*/` - Training checkpoints
- [ ] `comfyui_lora_model*/` - Trained models
- [ ] `__pycache__/` - Python cache
- [ ] `*.log` - Log files

### Create .gitignore
```bash
cat > .gitignore << 'EOF'
# Virtual environment
venv/
env/

# Dataset files (regenerate on server)
*.jsonl

# Training outputs
training_output*/
comfyui_lora_model*/
training_interrupted*/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Logs
*.log

# OS
.DS_Store
Thumbs.db
EOF
```

### Git Commands
```bash
git add .
git commit -m "Add server training pipeline with dataset merger and email notifications"
git push origin main
```

---

## 🖥️ Server Deployment

### 1. Initial Setup
```bash
# SSH to server
ssh user@your-server

# Clone repository
git clone <your-repo-url>
cd fine_tuning

# Run setup script
bash setup_server.sh
```

**Expected time**: 10-15 minutes

### 2. Verify Setup
```bash
# Check Python
python3 --version

# Check virtual environment
source venv/bin/activate

# Verify CPU optimization
python cpu_setup_server.py
```

**Expected output**: `CPU threads: 48`

### 3. Configure Email
```bash
# Set your email
export TRAINING_NOTIFICATION_EMAIL="your@email.com"

# Add to .bashrc for persistence
echo 'export TRAINING_NOTIFICATION_EMAIL="your@email.com"' >> ~/.bashrc

# Test email (optional)
python send_email_notification.py your@email.com 1.0 0.5 100
```

### 4. Test Mail Service (Optional)
```bash
# Check if mail service is running
systemctl status postfix
# or
systemctl status sendmail

# Test sending email
echo "Test email" | mail -s "Test" your@email.com
```

---

## 🚀 Start Training

### Option A: Full Automated Pipeline (Recommended)
```bash
# Run everything in one command
nohup bash run_full_training_pipeline.sh > pipeline.log 2>&1 &

# Monitor progress
tail -f pipeline.log
```

### Option B: Step by Step
```bash
# Step 1: Merge datasets (~15 min)
python merge_datasets.py

# Step 2: Format dataset (~3 min)
python format_combined_dataset.py

# Step 3: Train model (~3-5 hours)
nohup python train_model_server.py > training.log 2>&1 &

# Monitor
tail -f training.log
```

### Option C: Screen/Tmux (Interactive)
```bash
# Using screen
screen -S training
bash run_full_training_pipeline.sh
# Ctrl+A, D to detach

# Reattach later
screen -r training

# Or using tmux
tmux new -s training
bash run_full_training_pipeline.sh
# Ctrl+B, D to detach

# Reattach later
tmux attach -t training
```

---

## 📊 Monitoring

### Check Training Progress
```bash
# View log
tail -f training.log

# Check last 50 lines
tail -n 50 training.log

# Search for loss values
grep "loss" training.log | tail -20

# Check process
ps aux | grep train_model_server

# Monitor CPU
htop
```

### Expected Metrics
- **Loss**: Should decrease from ~2.0 to ~0.5-0.8
- **Grad norm**: Should be 0.5-2.0 (NOT 0.0!)
- **Speed**: ~1-2 seconds per step
- **CPU usage**: 100% across all 48 cores
- **Memory**: 8-10 GB

---

## ✅ After Training

### 1. Check Email
- [ ] Received completion notification
- [ ] Training time looks reasonable (3-5 hours)
- [ ] Final loss is good (0.5-0.8)

### 2. Test Model
```bash
python test_model.py
```

**Expected**: 3-4 out of 4 tests should pass with valid JSON

### 3. Download Model (if needed)
```bash
# From your local machine
scp -r user@server:/path/to/comfyui_lora_model_final ./
```

### 4. Verify Model Files
```bash
ls -lh comfyui_lora_model_final/
```

**Should contain**:
- `adapter_model.bin` (~4 MB)
- `adapter_config.json`
- Tokenizer files

---

## 🐛 Troubleshooting

### Training Not Starting
```bash
# Check if virtual environment is activated
which python
# Should show: /path/to/venv/bin/python

# Check dependencies
pip list | grep torch
pip list | grep transformers
```

### Email Not Sending
```bash
# Check mail service
systemctl status postfix

# Check mail logs
tail -f /var/log/mail.log

# Test mail command
echo "test" | mail -s "test" your@email.com

# Check environment variable
echo $TRAINING_NOTIFICATION_EMAIL
```

### Training Very Slow
```bash
# Verify CPU threads
python cpu_setup_server.py

# Check CPU usage
htop

# Check if other processes are running
ps aux | grep python
```

### Out of Memory
```bash
# Check memory usage
free -h

# Reduce batch size in train_model_server.py
# Change: per_device_train_batch_size=1
```

### Grad Norm = 0.0
This means no learning is happening. **Restart training from scratch**, don't resume from checkpoint.

---

## 📝 Final Checklist

### Before Leaving Server
- [ ] Training is running (check with `ps aux | grep train`)
- [ ] Log file is being written (`tail -f training.log` shows updates)
- [ ] Email is configured (`echo $TRAINING_NOTIFICATION_EMAIL`)
- [ ] Can disconnect safely (using nohup/screen/tmux)

### Expected Timeline
- [ ] Dataset merge: 15 minutes
- [ ] Dataset format: 3 minutes
- [ ] Training: 3-5 hours
- [ ] Email notification: Instant
- [ ] **Total: ~4-5 hours**

### Success Criteria
- [ ] Training completes without errors
- [ ] Final loss < 0.8
- [ ] Grad norm between 0.5-2.0 (not 0.0)
- [ ] Email received with summary
- [ ] Model files saved correctly
- [ ] Test script passes 3-4/4 tests

---

## 🎉 You're Done!

Once training completes and tests pass, you have a working ComfyUI workflow generator trained on 1000 real + synthetic examples!

**Next steps**:
1. Deploy to production
2. Integrate with your application
3. Consider training for more epochs if needed
4. Expand dataset with more examples

**Good luck!** 🚀

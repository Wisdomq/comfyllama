# ComfyLlama 🦙🎨

A fine-tuning pipeline for training **TinyLlama** to generate **ComfyUI workflow JSON** using LoRA (Low-Rank Adaptation). The project includes dataset creation, model training, CPU-optimized inference, and full deployment tooling for both local machines and remote servers.

---

## Overview

ComfyLlama teaches TinyLlama-1.1B to produce valid ComfyUI workflow graphs from natural-language descriptions. Rather than relying on a large GPU, the entire pipeline is engineered to run efficiently on CPU — including a server-optimized path targeting AMD EPYC 48-core hardware.

The training process is structured as a numbered phase pipeline:
- **Dataset creation** — synthetic ComfyUI workflow examples + real workflows from HuggingFace
- **LoRA fine-tuning** — parameter-efficient training on TinyLlama
- **Inference & testing** — load and test the adapted model
- **Deployment** — automated server setup and email notifications on completion

---

## Repository Structure

```
comfyllama/
├── create_comfyui_dataset.py       # Generate synthetic training examples from templates
├── explore_hf_dataset.py           # Explore real ComfyUI workflow datasets on HuggingFace
├── extract_workflows_from_hf.py    # Download and extract workflows from HuggingFace
├── merge_datasets.py               # Merge synthetic + real HuggingFace datasets
├── format_dataset.py               # Format dataset for TinyLlama chat template
├── format_combined_dataset.py      # Format merged dataset
├── validate_comfyui_dataset.py     # Validate dataset integrity
│
├── cpu_setup.py                    # CPU thread optimization for local training
├── cpu_setup_server.py             # CPU thread optimization for AMD EPYC (48 cores)
├── set_cpu_threads.ps1             # PowerShell script for Windows CPU thread config
├── verify_cpu_optimization.py      # Verify CPU settings are applied correctly
│
├── train_model.py                  # Local training script (~50 min, 500 examples × 3 epochs)
├── train_model_server.py           # Server-optimized training (AMD EPYC 7402P, 48 threads)
├── resume_training.py              # Resume training from a checkpoint
├── run_full_training_pipeline.sh   # End-to-end pipeline: merge → format → train
├── setup_server.sh                 # One-shot server environment setup
│
├── apply_lora.py                   # Merge LoRA adapters into the base model
├── load_model.py                   # Load and prepare model for inference
├── test_model.py                   # Test fine-tuned model on sample prompts
├── lora_visualization.py           # Visualize LoRA weight distributions
│
├── send_email_notification.py      # Email alert on training completion
├── test_install.py                 # Verify all dependencies are installed
│
├── SERVER_TRAINING_GUIDE.md        # Step-by-step server deployment guide
└── DEPLOYMENT_CHECKLIST.md        # Pre-deployment and post-training checklist
```

---

## Workflow Templates

The dataset creator supports three base ComfyUI workflow types:

| Template | Description |
|---|---|
| `basic_txt2img` | Text-to-image using KSampler + VAEDecode |
| `with_lora` | Text-to-image with LoRA model injection |
| `txt2video` | Text-to-video using VideoLinearCFGGuidance + VHS_VideoCombine |

---

## Training Configuration

### Local (CPU — laptop/desktop)

| Parameter | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| Target modules | `q_proj`, `v_proj` |
| Epochs | 3 |
| Batch size | 1 (effective: 8 with gradient accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Max sequence length | 1024 tokens |
| Expected time | ~50 minutes for 500 examples |

### Server (AMD EPYC 7402P — 48 cores)

The server path uses a larger batch size (`2`), 8 dataloader workers, and gradient clipping for stability. Expected training time is 3–5 hours on the combined 1,000-example dataset.

---

## Quick Start

### Local Training

```bash
# 1. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.38.0 datasets==2.18.0 accelerate==0.27.0 peft==0.9.0 trl==0.7.11
pip install sentencepiece protobuf scipy scikit-learn tqdm pillow

# 2. Generate synthetic dataset
python create_comfyui_dataset.py

# 3. Format for TinyLlama chat template
python format_dataset.py

# 4. Train
python train_model.py

# 5. Test the fine-tuned model
python test_model.py
```

### Server Deployment

```bash
# 1. SSH to server and clone the repo
git clone <your-repo-url> && cd comfyllama

# 2. Run automated setup (creates venv, installs deps)
bash setup_server.sh

# 3. Activate environment and configure email
source venv/bin/activate
export TRAINING_NOTIFICATION_EMAIL="your@email.com"

# 4. Run the full pipeline (merge → format → train)
nohup bash run_full_training_pipeline.sh > pipeline.log 2>&1 &

# 5. Monitor
tail -f pipeline.log
```

---

## Dataset Pipeline

```
create_comfyui_dataset.py     →  synthetic examples (500)
extract_workflows_from_hf.py  →  real HuggingFace workflows (500+)
merge_datasets.py             →  combined dataset (~1,000 examples)
format_combined_dataset.py    →  comfyui_dataset_combined_formatted.jsonl
```

Each training example follows the TinyLlama chat format:

```
<|system|>You are a ComfyUI workflow generator...</|system|>
<|user|>Generate a text-to-image workflow for a portrait photo...
Parameters: ...</|user|>
<|assistant|>{ "1": { "class_type": "KSampler", ... } }
```

---

## Model Output

After training, the LoRA adapters (~4 MB) are saved to `./comfyui_lora_model/` alongside the tokenizer. The adapter can be merged back into the base model with `apply_lora.py` for standalone deployment.

**Expected loss curve:** `~2.5 → ~0.8–1.5` over 3 epochs.

---

## Monitoring Training

```bash
# Watch live loss in log
tail -f training.log
grep "loss" training.log | tail -20

# Check CPU utilization
htop

# Check process
ps aux | grep train_model_server
```

**Healthy training signals:**
- Loss decreasing steadily (not stuck or 0)
- Grad norm in range `0.5–2.0` (a grad norm of `0.0` means no learning — restart from scratch)
- CPU usage at 100% across all cores
- Memory usage 8–10 GB

---

## Requirements

- Python 3.9+
- PyTorch (CPU build)
- `transformers` 4.38.0
- `peft` 0.9.0
- `trl` 0.7.11
- `datasets` 2.18.0
- `accelerate` 0.27.0
- 8–16 GB RAM (local), 32+ GB RAM (server)
- No GPU required

---

## Files to Exclude from Git

The `.gitignore` covers: `venv/`, `*.jsonl` dataset files, `training_output*/`, `comfyui_lora_model*/`, `__pycache__/`, `*.log`. These are either too large for version control or environment-specific and should be regenerated on the target machine.

---

## License

This project is unlicensed. All rights reserved to the author.

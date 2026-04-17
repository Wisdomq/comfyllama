"""
Load TinyLlama model for fine-tuning
This script demonstrates proper model loading for CPU training
"""

# IMPORTANT: Setup CPU optimization first
from cpu_setup import setup_cpu
setup_cpu(num_threads=16)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Correct Hugging Face model name (not Ollama format!)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("="*60)
print("LOADING TINYLLAMA MODEL")
print("="*60)

print("\n[1/2] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Added padding token")
else:
    print("  ✓ Padding token already exists")

print(f"  ✓ Tokenizer loaded")
print(f"  Vocabulary size: {len(tokenizer)}")

print("\n[2/2] Loading model...")
print("  This will take 1-2 minutes and download ~2 GB...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print(f"\n{'='*60}")
print("MODEL LOADED SUCCESSFULLY!")
print(f"{'='*60}")
print(f"\nModel: {model_name}")
print(f"Parameters: {model.num_parameters():,}")
print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

# Show model architecture summary
print(f"\nArchitecture:")
print(f"  Layers: {model.config.num_hidden_layers}")
print(f"  Hidden size: {model.config.hidden_size}")
print(f"  Attention heads: {model.config.num_attention_heads}")
print(f"  Vocabulary size: {model.config.vocab_size}")

print(f"\n{'='*60}")
print("READY FOR LORA APPLICATION")
print(f"{'='*60}")
print("\nNext step: Run apply_lora.py to add LoRA adapters")
print(f"{'='*60}")

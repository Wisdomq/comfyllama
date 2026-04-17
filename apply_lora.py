"""
Apply LoRA adapters to TinyLlama
This script demonstrates how to add trainable LoRA adapters to a frozen base model
"""

# IMPORTANT: Setup CPU optimization first
from cpu_setup import setup_cpu
setup_cpu(num_threads=16)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch

print("="*70)
print("APPLYING LORA TO TINYLLAMA")
print("="*70)

# Step 1: Load base model
print("\n[1/3] Loading base model...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

base_params = model.num_parameters()
print(f"✓ Base model loaded: {base_params:,} parameters")

# Step 2: Configure LoRA
print("\n[2/3] Configuring LoRA...")
print("\nLoRA Configuration:")
print("  Task: ComfyUI JSON workflow generation")
print("  Strategy: Minimal config for structural task")

lora_config = LoraConfig(
    r=8,                              # Rank: sufficient for JSON structure
    lora_alpha=16,                    # Scaling: 2 × r
    target_modules=["q_proj", "v_proj"],  # Minimal: only attention Q and V
    lora_dropout=0.05,                # Light regularization
    bias="none",                      # Don't train bias terms
    task_type=TaskType.CAUSAL_LM,     # Causal language modeling
)

print(f"\n  Rank (r): {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Target modules: {lora_config.target_modules}")
print(f"  Dropout: {lora_config.lora_dropout}")

# Step 3: Apply LoRA
print("\n[3/3] Applying LoRA adapters...")
model = get_peft_model(model, lora_config)

print("\n" + "="*70)
print("LORA APPLIED SUCCESSFULLY!")
print("="*70)

# Show trainable parameters
print("\nParameter Analysis:")
model.print_trainable_parameters()

# Calculate details
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
trainable_percent = (trainable_params / all_params) * 100

print(f"\nDetailed Breakdown:")
print(f"  Base model (frozen): {base_params:,} parameters")
print(f"  LoRA adapters (trainable): {trainable_params:,} parameters")
print(f"  Total parameters: {all_params:,}")
print(f"  Trainable percentage: {trainable_percent:.3f}%")

# Memory estimation
base_memory_gb = base_params * 4 / 1e9  # float32
lora_memory_gb = trainable_params * 4 / 1e9
optimizer_memory_gb = trainable_params * 8 / 1e9  # AdamW uses 2x
gradient_memory_gb = trainable_params * 4 / 1e9
activation_memory_gb = 1.5  # Approximate

total_memory_gb = base_memory_gb + lora_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb

print(f"\nMemory Estimation:")
print(f"  Base model: {base_memory_gb:.2f} GB")
print(f"  LoRA adapters: {lora_memory_gb:.3f} GB")
print(f"  Optimizer states: {optimizer_memory_gb:.3f} GB")
print(f"  Gradients: {gradient_memory_gb:.3f} GB")
print(f"  Activations: {activation_memory_gb:.2f} GB")
print(f"  ─────────────────────")
print(f"  Total estimated: {total_memory_gb:.2f} GB")

if total_memory_gb < 14:
    print(f"  Status: ✅ EXCELLENT - Plenty of headroom on 16 GB RAM")
elif total_memory_gb < 16:
    print(f"  Status: ✅ GOOD - Will fit comfortably")
else:
    print(f"  Status: ⚠️ WARNING - May be tight")

# Training time estimation
examples_per_hour = 250  # Approximate for TinyLlama on CPU with r=8
print(f"\nTraining Time Estimation:")
print(f"  Speed: ~{examples_per_hour} examples/hour")
print(f"  500 examples: ~{500/examples_per_hour:.1f} hours")
print(f"  1000 examples: ~{1000/examples_per_hour:.1f} hours")

# Show model architecture with LoRA
print("\n" + "="*70)
print("MODEL ARCHITECTURE WITH LORA")
print("="*70)
print(f"\nBase: TinyLlama-1.1B-Chat")
print(f"  Layers: {model.config.num_hidden_layers}")
print(f"  Hidden size: {model.config.hidden_size}")
print(f"  Attention heads: {model.config.num_attention_heads}")

print(f"\nLoRA Adapters:")
print(f"  Modules per layer: {len(lora_config.target_modules)}")
print(f"  Total adapters: {model.config.num_hidden_layers * len(lora_config.target_modules)}")
print(f"  Parameters per adapter: {model.config.hidden_size * lora_config.r * 2:,}")

print("\n" + "="*70)
print("READY FOR TRAINING!")
print("="*70)
print("\nNext steps:")
print("  1. Create your ComfyUI JSON dataset (Phase 4)")
print("  2. Format the dataset (Phase 5)")
print("  3. Run training (Phase 6)")
print("="*70)

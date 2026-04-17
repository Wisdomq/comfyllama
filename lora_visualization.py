"""
LoRA Visualization and Interactive Learning
Run this to see LoRA in action with real numbers
"""

from cpu_setup import setup_cpu
setup_cpu(num_threads=16)

import torch
import numpy as np

print("="*70)
print("LoRA INTERACTIVE VISUALIZATION")
print("="*70)

# Simulate a weight matrix from a transformer
print("\n" + "─"*70)
print("SCENARIO: One Attention Layer")
print("─"*70)

hidden_size = 4096  # Typical for 7B models
print(f"\nHidden size: {hidden_size}")

# Original weight matrix
W_size = (hidden_size, hidden_size)
W_params = hidden_size * hidden_size
W_memory_mb = W_params * 4 / 1e6  # 4 bytes per float32

print(f"\nOriginal Weight Matrix (W):")
print(f"  Shape: {W_size[0]} × {W_size[1]}")
print(f"  Parameters: {W_params:,}")
print(f"  Memory: {W_memory_mb:.2f} MB")

# LoRA matrices for different ranks
print("\n" + "─"*70)
print("LoRA CONFIGURATIONS")
print("─"*70)

ranks = [4, 8, 16, 32, 64]

print(f"\n{'Rank':<8} {'A Shape':<15} {'B Shape':<15} {'Params':<12} {'Memory':<10} {'% of W':<10}")
print("─"*70)

for r in ranks:
    A_shape = (hidden_size, r)
    B_shape = (r, hidden_size)
    
    lora_params = (hidden_size * r) + (r * hidden_size)
    lora_memory_mb = lora_params * 4 / 1e6
    percent_of_w = (lora_params / W_params) * 100
    
    print(f"r={r:<5} {str(A_shape):<15} {str(B_shape):<15} {lora_params:<12,} {lora_memory_mb:<10.2f} {percent_of_w:<10.3f}%")

# Full model calculation
print("\n" + "─"*70)
print("FULL MODEL CALCULATION (TinyLlama 1.1B)")
print("─"*70)

num_layers = 22  # TinyLlama has 22 layers
projections_per_layer = 2  # q_proj and v_proj

print(f"\nModel: TinyLlama-1.1B")
print(f"Layers: {num_layers}")
print(f"Target modules: q_proj, v_proj ({projections_per_layer} per layer)")

print(f"\n{'Rank':<8} {'Total LoRA Params':<20} {'% of 1.1B':<15} {'Memory (MB)':<15}")
print("─"*70)

for r in ranks:
    params_per_projection = (hidden_size * r) + (r * hidden_size)
    total_lora_params = params_per_projection * projections_per_layer * num_layers
    percent_of_model = (total_lora_params / 1.1e9) * 100
    memory_mb = total_lora_params * 4 / 1e6
    
    print(f"r={r:<5} {total_lora_params:<20,} {percent_of_model:<15.3f}% {memory_mb:<15.2f}")

# Memory comparison
print("\n" + "─"*70)
print("MEMORY COMPARISON: Full Fine-Tuning vs LoRA (r=8)")
print("─"*70)

model_size_gb = 4.4  # TinyLlama in float32

print("\nFull Fine-Tuning:")
print(f"  Model weights: {model_size_gb:.1f} GB")
print(f"  Optimizer states (AdamW): {model_size_gb * 2:.1f} GB")
print(f"  Gradients: {model_size_gb:.1f} GB")
print(f"  Activations: ~1.5 GB")
print(f"  ─────────────────────")
print(f"  TOTAL: ~{model_size_gb * 4 + 1.5:.1f} GB")
print(f"  Status on 16 GB RAM: ❌ TOO MUCH")

r = 8
params_per_projection = (hidden_size * r) + (r * hidden_size)
total_lora_params = params_per_projection * projections_per_layer * num_layers
lora_size_gb = total_lora_params * 4 / 1e9

print(f"\nLoRA Fine-Tuning (r={r}):")
print(f"  Model weights (frozen): {model_size_gb:.1f} GB")
print(f"  LoRA adapters: {lora_size_gb:.3f} GB")
print(f"  Optimizer states (LoRA only): {lora_size_gb * 2:.3f} GB")
print(f"  Gradients (LoRA only): {lora_size_gb:.3f} GB")
print(f"  Activations: ~1.5 GB")
print(f"  ─────────────────────")
total_lora_memory = model_size_gb + (lora_size_gb * 4) + 1.5
print(f"  TOTAL: ~{total_lora_memory:.1f} GB")
print(f"  Status on 16 GB RAM: ✅ FITS COMFORTABLY")

memory_savings = ((model_size_gb * 4 + 1.5) - total_lora_memory) / (model_size_gb * 4 + 1.5) * 100
print(f"\n  Memory savings: {memory_savings:.1f}%")

# Interactive: Calculate for your configuration
print("\n" + "="*70)
print("INTERACTIVE: Calculate Your Configuration")
print("="*70)

print("\nLet's calculate memory for YOUR chosen configuration:")
print("(Press Enter to use defaults)")

try:
    rank_input = input(f"\nEnter rank (default 8): ").strip()
    chosen_rank = int(rank_input) if rank_input else 8
    
    modules_input = input("Target modules - enter number:\n  1 = q_proj, v_proj (2 modules)\n  2 = q_proj, k_proj, v_proj, o_proj (4 modules)\n  3 = All 7 modules\nChoice (default 1): ").strip()
    
    if modules_input == "2":
        num_modules = 4
        module_names = "q_proj, k_proj, v_proj, o_proj"
    elif modules_input == "3":
        num_modules = 7
        module_names = "All attention + MLP"
    else:
        num_modules = 2
        module_names = "q_proj, v_proj"
    
    # Calculate
    params_per_projection = (hidden_size * chosen_rank) + (chosen_rank * hidden_size)
    total_params = params_per_projection * num_modules * num_layers
    lora_mem_gb = total_params * 4 / 1e9
    total_mem = model_size_gb + (lora_mem_gb * 4) + 1.5
    
    print(f"\n{'─'*70}")
    print("YOUR CONFIGURATION:")
    print(f"{'─'*70}")
    print(f"  Rank: {chosen_rank}")
    print(f"  Target modules: {module_names} ({num_modules} per layer)")
    print(f"  Total LoRA parameters: {total_params:,}")
    print(f"  Percentage of model: {(total_params / 1.1e9) * 100:.3f}%")
    print(f"  Total memory needed: ~{total_mem:.1f} GB")
    
    if total_mem < 14:
        print(f"  Status: ✅ EXCELLENT - Plenty of headroom")
    elif total_mem < 16:
        print(f"  Status: ✅ GOOD - Will fit on your system")
    else:
        print(f"  Status: ⚠️ WARNING - May be tight, consider reducing rank or modules")
    
    # Training time estimate
    base_time_hours = 3.5  # Base time for r=8, 2 modules, 1000 examples
    time_multiplier = (chosen_rank / 8) * (num_modules / 2)
    estimated_time = base_time_hours * time_multiplier
    
    print(f"\n  Estimated training time (1000 examples): ~{estimated_time:.1f} hours")
    
except KeyboardInterrupt:
    print("\n\nSkipped interactive section")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. LoRA reduces trainable parameters by 95-99%
2. Memory savings allow CPU training of 1-2B models
3. Rank controls capacity: higher = more learning but slower
4. Target modules control coverage: more = better quality but slower
5. For CPU: Start with r=8, 2 modules, then adjust based on results
""")
print("="*70)

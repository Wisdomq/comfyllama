"""
Verify CPU optimization settings
Run this after setting environment variables to confirm they're working
"""

import torch
import os
import platform

print("="*60)
print("CPU OPTIMIZATION VERIFICATION")
print("="*60)

# System info
print(f"\nSystem: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")

# Environment variables
print("\n" + "-"*60)
print("Environment Variables:")
print("-"*60)
env_vars = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", 
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS"
]

all_set = True
for var in env_vars:
    value = os.environ.get(var, "NOT SET")
    status = "✓" if value != "NOT SET" else "✗"
    print(f"{status} {var}: {value}")
    if value == "NOT SET":
        all_set = False

# PyTorch settings
print("\n" + "-"*60)
print("PyTorch Configuration:")
print("-"*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU threads: {torch.get_num_threads()}")

# Analysis
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

expected_threads = 16  # For Ryzen 7 5700U
actual_threads = torch.get_num_threads()

if actual_threads == expected_threads:
    print(f"✓ OPTIMAL: Using all {actual_threads} threads")
    print("  Your CPU is fully utilized for training")
elif actual_threads < expected_threads:
    print(f"⚠ SUBOPTIMAL: Using {actual_threads}/{expected_threads} threads")
    print(f"  Training will be ~{expected_threads/actual_threads:.1f}x slower than optimal")
    print("\nTo fix:")
    print("  1. Run: .\\set_cpu_threads.ps1")
    print("  2. Then run this script again to verify")
else:
    print(f"✓ Using {actual_threads} threads")

if not all_set:
    print("\n⚠ WARNING: Some environment variables not set")
    print("  Run set_cpu_threads.ps1 before training")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if actual_threads < expected_threads:
    print("Before each training session:")
    print("  1. Open PowerShell in this directory")
    print("  2. Run: .\\set_cpu_threads.ps1")
    print("  3. Activate venv: .\\venv\\Scripts\\Activate.ps1")
    print("  4. Run training script")
else:
    print("✓ Your system is optimized for CPU training!")
    print("  You can proceed with training")

print("="*60)

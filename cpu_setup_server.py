"""
CPU Optimization for AMD EPYC 7402P Server
48 threads (24 cores × 2 threads/core)
"""

import os

# Set environment variables for maximum CPU utilization
# AMD EPYC 7402P has 48 threads
os.environ["OMP_NUM_THREADS"] = "48"
os.environ["MKL_NUM_THREADS"] = "48"
os.environ["OPENBLAS_NUM_THREADS"] = "48"
os.environ["VECLIB_MAXIMUM_THREADS"] = "48"
os.environ["NUMEXPR_NUM_THREADS"] = "48"

# Import torch AFTER setting environment variables
import torch

# Set PyTorch to use all 48 threads
torch.set_num_threads(48)

# Verify
if __name__ == "__main__":
    print("="*70)
    print("CPU OPTIMIZATION - AMD EPYC 7402P")
    print("="*70)
    print(f"\nEnvironment variables:")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS')}")
    print(f"\nPyTorch configuration:")
    print(f"  Version: {torch.__version__}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print("\n✓ CPU optimized for 48 threads")
    print("="*70)

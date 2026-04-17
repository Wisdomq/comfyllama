"""
CPU Optimization Setup Module
Import this at the start of every training script to maximize CPU utilization

Usage:
    from cpu_setup import setup_cpu
    setup_cpu()
"""

import os
import sys

def setup_cpu(num_threads=16):
    """
    Configure CPU for optimal training performance
    
    Args:
        num_threads: Number of threads to use (default: 16 for Ryzen 7 5700U)
    
    Returns:
        dict: Configuration status
    """
    
    # Set environment variables BEFORE importing torch
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    
    # Import torch and set threads directly
    import torch
    torch.set_num_threads(num_threads)
    
    # Verify
    actual_threads = torch.get_num_threads()
    
    status = {
        "requested_threads": num_threads,
        "actual_threads": actual_threads,
        "success": actual_threads == num_threads,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    
    return status

def print_cpu_status():
    """Print detailed CPU configuration status"""
    import torch
    
    print("="*60)
    print("CPU CONFIGURATION STATUS")
    print("="*60)
    
    env_vars = {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "Not set"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "Not set"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "Not set"),
    }
    
    print("\nEnvironment Variables:")
    for var, value in env_vars.items():
        status = "✓" if value != "Not set" else "✗"
        print(f"  {status} {var}: {value}")
    
    print(f"\nPyTorch Configuration:")
    print(f"  Version: {torch.__version__}")
    print(f"  CPU Threads: {torch.get_num_threads()}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.get_num_threads() == 16:
        print(f"\n✓ OPTIMAL: Using all 16 threads")
    else:
        print(f"\n⚠ Using {torch.get_num_threads()}/16 threads")
    
    print("="*60)

if __name__ == "__main__":
    # Test the setup
    print("Testing CPU optimization setup...\n")
    status = setup_cpu()
    print_cpu_status()
    
    if status["success"]:
        print("\n✓ CPU optimization successful!")
    else:
        print(f"\n⚠ Warning: Requested {status['requested_threads']} threads, got {status['actual_threads']}")

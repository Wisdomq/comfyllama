"""
Test installation and CPU optimization
Run this to verify your environment is set up correctly
"""

# IMPORTANT: Import cpu_setup FIRST, before any other imports
from cpu_setup import setup_cpu, print_cpu_status

# Setup CPU optimization
print("Setting up CPU optimization...")
status = setup_cpu(num_threads=16)
print()

# Now import other packages
import torch
import transformers
import peft
import trl

# Print detailed status
print_cpu_status()

# Print package versions
print("\nPackage Versions:")
print(f"  Transformers: {transformers.__version__}")
print(f"  PEFT: {peft.__version__}")
print(f"  TRL: {trl.__version__}")

# Final verdict
print("\n" + "="*60)
if status["success"] and status["actual_threads"] == 16:
    print("✓ ALL CHECKS PASSED - Ready for training!")
    print("  Your CPU is optimized for maximum performance")
else:
    print("⚠ OPTIMIZATION INCOMPLETE")
    print(f"  Expected 16 threads, got {status['actual_threads']}")
print("="*60)
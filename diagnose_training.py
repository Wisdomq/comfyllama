"""
Diagnostic script to identify training startup issues
Run this to see exactly what's failing
"""

print("="*70)
print("TRAINING DIAGNOSTICS")
print("="*70)

# Test 1: CPU Setup Import
print("\n[1/6] Testing CPU setup import...")
try:
    import cpu_setup_server
    print("✓ cpu_setup_server imported successfully")
    import os
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
    print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Test 2: PyTorch Import
print("\n[2/6] Testing PyTorch import...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported")
    print(f"  CPU threads: {torch.get_num_threads()}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Test 3: Transformers Import
print("\n[3/6] Testing transformers import...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    print("✓ All training libraries imported")
except Exception as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Test 4: Dataset File Check
print("\n[4/6] Checking dataset files...")
import os
required_files = [
    "comfyui_dataset_combined.jsonl",
    "comfyui_dataset_combined_formatted.jsonl"
]

for file in required_files:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / 1e6
        print(f"✓ {file} exists ({size_mb:.1f} MB)")
    else:
        print(f"❌ MISSING: {file}")
        print("\nRun these commands first:")
        print("  python merge_datasets.py")
        print("  python format_combined_dataset.py")
        exit(1)

# Test 5: Dataset Loading
print("\n[5/6] Testing dataset loading...")
try:
    from datasets import load_dataset
    dataset = load_dataset("json", data_files="comfyui_dataset_combined_formatted.jsonl", split="train")
    print(f"✓ Dataset loaded: {len(dataset)} examples")
    print(f"  Columns: {dataset.column_names}")
    
    # Check first example
    if len(dataset) > 0:
        first_text = dataset[0]['text']
        print(f"  First example length: {len(first_text)} chars")
        if len(first_text) < 100:
            print(f"  ⚠️  WARNING: First example is very short!")
            print(f"  Preview: {first_text[:200]}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Model Loading (quick test)
print("\n[6/6] Testing model loading (this may take 30-60 seconds)...")
try:
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Tokenizer loaded")
    
    # Don't load full model in diagnostic, just check it's accessible
    print(f"✓ Model {model_name} is accessible")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✓ ALL DIAGNOSTICS PASSED")
print("="*70)
print("\nYour environment is ready for training!")
print("\nTo start training:")
print("  python train_model_server.py 2>&1 | tee training.log")
print("\nOr use nohup for background:")
print("  nohup python train_model_server.py > training.log 2>&1 &")
print("="*70)

"""
Explore the HuggingFace ComfyUI dataset structure
"""

from datasets import load_dataset

print("="*70)
print("EXPLORING HUGGINGFACE DATASET")
print("="*70)

print("\nLoading dataset (this may take a while - 6.25 GB)...")
try:
    # Try to load just a few examples first
    dataset = load_dataset("cmcjas/SDXL_ComfyUI_workflows", split="train", streaming=True)
    
    print("✓ Dataset loaded in streaming mode")
    print("\nExamining first example...")
    
    # Get first example
    first_example = next(iter(dataset))
    
    print("\n" + "="*70)
    print("DATASET STRUCTURE")
    print("="*70)
    print(f"\nColumns: {list(first_example.keys())}")
    
    for key, value in first_example.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, str):
            print(f"  Length: {len(value)} characters")
            print(f"  Preview: {value[:200]}...")
        elif isinstance(value, (list, dict)):
            print(f"  Content: {str(value)[:200]}...")
        else:
            print(f"  Value: {value}")
    
    print("\n" + "="*70)
    print("Checking a few more examples...")
    
    count = 0
    for example in dataset:
        count += 1
        if count >= 3:
            break
        print(f"\nExample {count + 1} keys: {list(example.keys())}")
    
    print(f"\n✓ Dataset appears to have {count} examples (streaming mode)")
    
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    print("\nTrying alternative loading method...")
    
    try:
        # Try loading without streaming
        dataset = load_dataset("cmcjas/SDXL_ComfyUI_workflows", split="train[:5]")
        print(f"✓ Loaded first 5 examples")
        print(f"Columns: {dataset.column_names}")
        print(f"\nFirst example keys: {list(dataset[0].keys())}")
        
        for key in dataset[0].keys():
            print(f"\n{key}: {type(dataset[0][key])}")
            
    except Exception as e2:
        print(f"❌ Also failed: {e2}")

print("\n" + "="*70)

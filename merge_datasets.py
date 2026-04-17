"""
Merge HuggingFace ComfyUI workflows with our synthetic dataset
Extracts workflows from PNG metadata and combines with existing data
"""

from datasets import load_dataset
import json
from tqdm import tqdm

print("="*70)
print("MERGING COMFYUI DATASETS")
print("="*70)

# Step 1: Load existing synthetic dataset
print("\n[1/4] Loading existing synthetic dataset...")
with open("comfyui_dataset.jsonl", "r") as f:
    synthetic_data = [json.loads(line) for line in f]

print(f"✓ Loaded {len(synthetic_data)} synthetic examples")

# Step 2: Load HuggingFace dataset and extract workflows
print("\n[2/4] Loading HuggingFace dataset (500 real workflows)...")
print("This will download ~6.25 GB of data...")

hf_dataset = load_dataset("cmcjas/SDXL_ComfyUI_workflows", split="train")
print(f"✓ Loaded {len(hf_dataset)} images with workflows")

# Step 3: Extract workflows from images
print("\n[3/4] Extracting workflows from PNG metadata...")

extracted_workflows = []
failed_extractions = 0

for idx, example in enumerate(tqdm(hf_dataset, desc="Extracting")):
    try:
        image = example['image']
        
        # Extract workflow JSON from PNG metadata
        if 'workflow' in image.info:
            workflow_json = image.info['workflow']
            workflow = json.loads(workflow_json)
            
            # Convert to our training format
            # We'll create a generic instruction since we don't have the original prompt
            training_example = {
                "instruction": "Generate a ComfyUI workflow for image generation",
                "input": f"Create a workflow with {len(workflow.get('nodes', []))} nodes",
                "output": json.dumps(workflow, separators=(',', ':'))
            }
            
            extracted_workflows.append(training_example)
        else:
            failed_extractions += 1
            
    except Exception as e:
        failed_extractions += 1
        print(f"\n⚠️  Failed to extract workflow {idx}: {e}")

print(f"\n✓ Extracted {len(extracted_workflows)} workflows")
print(f"✗ Failed: {failed_extractions}")

# Step 4: Combine datasets
print("\n[4/4] Combining datasets...")

combined_data = synthetic_data + extracted_workflows

print(f"\n✓ Combined dataset:")
print(f"  Synthetic examples: {len(synthetic_data)}")
print(f"  Real workflows: {len(extracted_workflows)}")
print(f"  Total: {len(combined_data)}")

# Save combined dataset
output_file = "comfyui_dataset_combined.jsonl"
with open(output_file, "w") as f:
    for example in combined_data:
        f.write(json.dumps(example) + "\n")

print(f"\n✓ Saved to: {output_file}")

# Statistics
print("\n" + "="*70)
print("DATASET STATISTICS")
print("="*70)

# Count instruction types
instruction_counts = {}
for example in combined_data:
    inst = example['instruction']
    instruction_counts[inst] = instruction_counts.get(inst, 0) + 1

print("\nInstruction distribution:")
for inst, count in sorted(instruction_counts.items(), key=lambda x: -x[1]):
    percentage = (count / len(combined_data)) * 100
    print(f"  {inst[:60]}...: {count} ({percentage:.1f}%)")

# Output length statistics
import numpy as np
output_lengths = [len(ex['output']) for ex in combined_data]
print(f"\nOutput length (characters):")
print(f"  Mean: {np.mean(output_lengths):.0f}")
print(f"  Median: {np.median(output_lengths):.0f}")
print(f"  Min: {np.min(output_lengths)}")
print(f"  Max: {np.max(output_lengths)}")

print("\n" + "="*70)
print("NEXT STEP")
print("="*70)
print("Format the combined dataset:")
print("  python format_combined_dataset.py")
print("="*70)

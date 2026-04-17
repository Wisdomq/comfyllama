"""
Format the combined dataset with TinyLlama chat template
"""

import cpu_setup
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

print("="*70)
print("FORMATTING COMBINED DATASET")
print("="*70)

# Load tokenizer
print("\n[1/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Tokenizer loaded")

# Load combined dataset
print("\n[2/3] Loading combined dataset...")
dataset = load_dataset("json", data_files="comfyui_dataset_combined.jsonl", split="train")
print(f"✓ Loaded {len(dataset)} examples")

def format_instruction(example):
    """Apply TinyLlama chat template"""
    if example['input']:
        user_message = f"{example['instruction']}\n\nParameters:\n{example['input']}"
    else:
        user_message = example['instruction']
    
    formatted_text = f"""<|system|>
You are a ComfyUI workflow generator. Generate valid ComfyUI workflow JSON based on user requirements.</|system|>
<|user|>
{user_message}</|user|>
<|assistant|>
{example['output']}</|assistant|>"""
    
    return {"text": formatted_text}

# Format dataset
print("\n[3/3] Applying chat template...")
formatted_dataset = dataset.map(
    format_instruction,
    remove_columns=dataset.column_names,
    desc="Formatting"
)

print(f"✓ Formatted {len(formatted_dataset)} examples")

# Analyze token lengths
print("\nAnalyzing token lengths...")
token_lengths = []

for i, example in enumerate(formatted_dataset):
    tokens = tokenizer.encode(example["text"])
    token_lengths.append(len(tokens))
    if (i + 1) % 100 == 0:
        print(f"  Analyzed {i + 1}/{len(formatted_dataset)}")

token_lengths = np.array(token_lengths)

print("\n" + "="*70)
print("TOKEN LENGTH ANALYSIS")
print("="*70)
print(f"Mean: {np.mean(token_lengths):.0f}")
print(f"Median: {np.median(token_lengths):.0f}")
print(f"Min: {np.min(token_lengths)}")
print(f"Max: {np.max(token_lengths)}")
print(f"Std: {np.std(token_lengths):.0f}")

percentiles = [25, 50, 75, 90, 95, 99]
print("\nPercentiles:")
for p in percentiles:
    print(f"  {p}th: {np.percentile(token_lengths, p):.0f}")

# Check for very long sequences
for max_len in [512, 1024, 2048]:
    over = np.sum(token_lengths > max_len)
    pct = (over / len(token_lengths)) * 100
    print(f"\nExamples > {max_len} tokens: {over} ({pct:.1f}%)")

# Recommend max_seq_length
if np.percentile(token_lengths, 95) <= 512:
    recommended = 512
elif np.percentile(token_lengths, 95) <= 1024:
    recommended = 1024
else:
    recommended = 2048

print(f"\n✓ Recommended max_seq_length: {recommended}")

# Save
output_file = "comfyui_dataset_combined_formatted.jsonl"
with open(output_file, "w") as f:
    for example in formatted_dataset:
        f.write(json.dumps(example) + "\n")

print(f"\n✓ Saved to: {output_file}")

print("\n" + "="*70)
print("READY FOR TRAINING")
print("="*70)
print(f"Dataset: {output_file}")
print(f"Examples: {len(formatted_dataset)}")
print(f"Recommended max_seq_length: {recommended}")
print("="*70)

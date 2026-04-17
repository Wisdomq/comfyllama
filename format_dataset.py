"""
Dataset Formatter for TinyLlama
Applies the TinyLlama chat template to the ComfyUI dataset
"""

import json
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Import CPU optimization first
import cpu_setup

print("="*70)
print("DATASET FORMATTER - PHASE 5")
print("="*70)

# Load tokenizer
print("\n[1/4] Loading TinyLlama tokenizer...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Added padding token")

print(f"✓ Tokenizer loaded")
print(f"  Vocab size: {len(tokenizer)}")
print(f"  EOS token: {tokenizer.eos_token}")
print(f"  PAD token: {tokenizer.pad_token}")

# Load dataset
print("\n[2/4] Loading dataset...")
dataset = load_dataset("json", data_files="comfyui_dataset.jsonl", split="train")
print(f"✓ Loaded {len(dataset)} examples")

def format_instruction(example):
    """
    Convert example to TinyLlama chat format
    
    TinyLlama uses this format:
    <|system|>
    System message</|system|>
    <|user|>
    User message</|user|>
    <|assistant|>
    Assistant response</|assistant|>
    """
    
    # Combine instruction and input for the user message
    if example['input']:
        user_message = f"{example['instruction']}\n\nParameters:\n{example['input']}"
    else:
        user_message = example['instruction']
    
    # Apply TinyLlama chat template
    formatted_text = f"""<|system|>
You are a ComfyUI workflow generator. Generate valid ComfyUI workflow JSON based on user requirements.</|system|>
<|user|>
{user_message}</|user|>
<|assistant|>
{example['output']}</|assistant|>"""
    
    return {"text": formatted_text}

# Apply formatting
print("\n[3/4] Applying TinyLlama chat template...")
formatted_dataset = dataset.map(
    format_instruction,
    remove_columns=dataset.column_names,
    desc="Formatting examples"
)

print(f"✓ Formatted {len(formatted_dataset)} examples")

# Analyze token lengths
print("\n[4/4] Analyzing token lengths...")
token_lengths = []

for i, example in enumerate(formatted_dataset):
    tokens = tokenizer.encode(example["text"])
    token_lengths.append(len(tokens))
    
    if (i + 1) % 100 == 0:
        print(f"  Analyzed {i + 1}/{len(formatted_dataset)} examples")

token_lengths = np.array(token_lengths)

print("\n" + "="*70)
print("TOKEN LENGTH ANALYSIS")
print("="*70)
print(f"Mean tokens: {np.mean(token_lengths):.0f}")
print(f"Median tokens: {np.median(token_lengths):.0f}")
print(f"Min tokens: {np.min(token_lengths)}")
print(f"Max tokens: {np.max(token_lengths)}")
print(f"Std deviation: {np.std(token_lengths):.0f}")

# Distribution
print("\nToken length distribution:")
percentiles = [25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = np.percentile(token_lengths, p)
    print(f"  {p}th percentile: {value:.0f} tokens")

# Check for long sequences
max_seq_length = 512
long_sequences = np.sum(token_lengths > max_seq_length)
if long_sequences > 0:
    percentage = (long_sequences / len(token_lengths)) * 100
    print(f"\n⚠️  WARNING: {long_sequences} examples ({percentage:.1f}%) exceed {max_seq_length} tokens")
    print(f"   Recommendation: Use max_seq_length=1024 in training")
else:
    print(f"\n✓ All examples fit within {max_seq_length} tokens")

# Save formatted dataset
print("\n" + "="*70)
print("SAVING FORMATTED DATASET")
print("="*70)

output_file = "comfyui_dataset_formatted.jsonl"
with open(output_file, "w") as f:
    for example in formatted_dataset:
        f.write(json.dumps(example) + "\n")

print(f"✓ Saved to: {output_file}")

# Preview examples
print("\n" + "="*70)
print("PREVIEW - First Example")
print("="*70)
print(formatted_dataset[0]["text"][:500] + "...")
print("\n[... truncated for display ...]")

print("\n" + "="*70)
print("PREVIEW - Token Breakdown")
print("="*70)
example = formatted_dataset[0]["text"]
tokens = tokenizer.encode(example)
print(f"Total tokens: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")
print(f"Decoded: {tokenizer.decode(tokens[:20])}")

print("\n" + "="*70)
print("✓ PHASE 5 COMPLETE - DATASET FORMATTED")
print("="*70)
print("\nYour dataset is now ready for training!")
print("\nNext steps:")
print("  1. Review the formatted dataset")
print("  2. Note the max token length for training config")
print("  3. Proceed to Phase 6: First Training Run")
print("\nKey information for training:")
print(f"  • Dataset file: {output_file}")
print(f"  • Number of examples: {len(formatted_dataset)}")
print(f"  • Recommended max_seq_length: {1024 if long_sequences > 0 else 512}")
print(f"  • Average tokens per example: {np.mean(token_lengths):.0f}")
print("="*70)

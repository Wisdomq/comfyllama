"""
Extract ComfyUI workflows from HuggingFace dataset images
ComfyUI embeds workflow JSON in PNG metadata
"""

from datasets import load_dataset
from PIL import Image
import json

print("="*70)
print("EXTRACTING WORKFLOWS FROM HUGGINGFACE DATASET")
print("="*70)

print("\nLoading dataset in streaming mode...")
dataset = load_dataset("cmcjas/SDXL_ComfyUI_workflows", split="train", streaming=True)

print("✓ Dataset loaded")
print("\nExtracting workflow from first image...")

# Get first image
first_example = next(iter(dataset))
image = first_example['image']

print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")

# ComfyUI stores workflow in PNG metadata under 'workflow' or 'prompt' keys
print("\nChecking PNG metadata...")
print(f"Available metadata keys: {list(image.info.keys())}")

# Try to extract workflow
workflow_found = False

if 'workflow' in image.info:
    print("\n✓ Found 'workflow' in metadata")
    workflow_json = image.info['workflow']
    print(f"Workflow length: {len(workflow_json)} characters")
    print(f"First 200 chars: {workflow_json[:200]}...")
    workflow_found = True
    
    # Try to parse it
    try:
        workflow = json.loads(workflow_json)
        print(f"\n✓ Valid JSON!")
        print(f"Workflow keys: {list(workflow.keys())}")
        if 'nodes' in workflow:
            print(f"Number of nodes: {len(workflow['nodes'])}")
        if 'links' in workflow:
            print(f"Number of links: {len(workflow['links'])}")
    except json.JSONDecodeError as e:
        print(f"\n✗ Invalid JSON: {e}")

if 'prompt' in image.info:
    print("\n✓ Found 'prompt' in metadata")
    prompt_json = image.info['prompt']
    print(f"Prompt length: {len(prompt_json)} characters")
    print(f"First 200 chars: {prompt_json[:200]}...")
    workflow_found = True

if not workflow_found:
    print("\n⚠️  No workflow metadata found in standard keys")
    print("Available keys:", list(image.info.keys()))
    print("\nChecking all metadata:")
    for key, value in image.info.items():
        print(f"\n{key}:")
        if isinstance(value, str):
            print(f"  Length: {len(value)}")
            print(f"  Preview: {value[:100]}...")

print("\n" + "="*70)
print("CHECKING MULTIPLE EXAMPLES")
print("="*70)

count = 0
workflows_extracted = 0

for example in dataset:
    count += 1
    if count > 5:  # Check first 5
        break
    
    image = example['image']
    has_workflow = 'workflow' in image.info or 'prompt' in image.info
    
    if has_workflow:
        workflows_extracted += 1
        print(f"✓ Example {count}: Has workflow metadata")
    else:
        print(f"✗ Example {count}: No workflow metadata")

print(f"\n{workflows_extracted}/{count} examples have extractable workflows")

print("\n" + "="*70)

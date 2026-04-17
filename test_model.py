"""
Test the Fine-Tuned ComfyUI Workflow Generator
Load and test the LoRA-adapted model
"""

# Import CPU optimization first
import cpu_setup

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

print("="*70)
print("TESTING FINE-TUNED COMFYUI WORKFLOW GENERATOR")
print("="*70)

# Load base model
print("\n[1/3] Loading base model...")
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
print("✓ Base model loaded")

# Load LoRA adapters
print("\n[2/3] Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, "./training_interrupted_2")
print("✓ LoRA adapters loaded")
print("✓ Model ready for inference!\n")

def generate_workflow(instruction, parameters, max_tokens=800, temperature=0.7):
    """
    Generate a ComfyUI workflow based on instruction and parameters
    
    Args:
        instruction: What type of workflow to generate
        parameters: Specific parameters for the workflow
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.7 = balanced, 0.1 = focused, 1.0 = creative)
    
    Returns:
        Generated workflow JSON string
    """
    
    # Format prompt using TinyLlama chat template
    prompt = f"""<|system|>
You are a ComfyUI workflow generator. Generate valid ComfyUI workflow JSON based on user requirements.</|system|>
<|user|>
{instruction}

Parameters:
{parameters}</|user|>
<|assistant|>
"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Add stopping criteria to ensure complete JSON
            repetition_penalty=1.1,  # Discourage repetition
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response

def validate_json(json_string):
    """Check if the generated output is valid JSON"""
    try:
        parsed = json.loads(json_string)
        return True, parsed
    except json.JSONDecodeError as e:
        return False, str(e)

# Test cases
print("="*70)
print("[3/3] RUNNING TEST CASES")
print("="*70)

test_cases = [
    {
        "name": "Text-to-Image (Basic)",
        "instruction": "Generate a ComfyUI workflow for text-to-image generation",
        "parameters": "Model: sd-v1-5, Prompt: 'a beautiful sunset over mountains', Negative: 'blurry, low quality', Sampler: euler, Scheduler: normal, Steps: 20, CFG: 7, Size: 512x512"
    },
    {
        "name": "Text-to-Image (with LoRA)",
        "instruction": "Create a ComfyUI workflow with LoRA",
        "parameters": "Base model: sdxl-base, LoRA: detail-tweaker, LoRA strength: 0.8, Prompt: 'a futuristic cityscape', Negative: 'distorted', Sampler: dpmpp_2m, Steps: 30, CFG: 8, Size: 1024x1024"
    },
    {
        "name": "Image-to-Image",
        "instruction": "Generate a ComfyUI workflow for image-to-image transformation",
        "parameters": "Input image: input.png, Model: sd-v2-1, Prompt: 'transform into oil painting style', Negative: 'blurry', Sampler: euler_a, Scheduler: karras, Steps: 25, CFG: 7, Denoise: 0.75"
    },
    {
        "name": "Text-to-Video",
        "instruction": "Generate a ComfyUI workflow for text-to-video generation",
        "parameters": "Model: svd-v1, Prompt: 'waves crashing on a beach', Negative: 'static, frozen', Sampler: euler, Scheduler: normal, Steps: 20, CFG: 7, Size: 512x512, Frames: 24, FPS: 12"
    },
]

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}/{len(test_cases)}: {test['name']}")
    print(f"{'='*70}")
    print(f"\nInstruction: {test['instruction']}")
    print(f"Parameters: {test['parameters'][:80]}...")
    
    print("\nGenerating workflow...")
    workflow = generate_workflow(test['instruction'], test['parameters'])
    
    print(f"\nGenerated output ({len(workflow)} characters):")
    print(workflow[:200] + "..." if len(workflow) > 200 else workflow)
    
    # Validate JSON
    is_valid, result = validate_json(workflow)
    
    if is_valid:
        print("\n✓ Valid JSON!")
        print(f"  Nodes: {len(result.get('nodes', []))}")
        print(f"  Links: {len(result.get('links', []))}")
        status = "PASS"
    else:
        print(f"\n✗ Invalid JSON: {result}")
        status = "FAIL"
    
    results.append({
        "test": test['name'],
        "status": status,
        "valid_json": is_valid,
        "output_length": len(workflow)
    })

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

passed = sum(1 for r in results if r['status'] == 'PASS')
total = len(results)

print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
print("\nDetailed results:")
for r in results:
    status_icon = "✓" if r['status'] == 'PASS' else "✗"
    print(f"  {status_icon} {r['test']}: {r['status']}")
    print(f"     Valid JSON: {r['valid_json']}, Output length: {r['output_length']} chars")

print("\n" + "="*70)
print("EVALUATION")
print("="*70)

if passed == total:
    print("🎉 Excellent! All tests passed.")
    print("   Your model generates valid ComfyUI workflows.")
elif passed >= total * 0.75:
    print("✓ Good! Most tests passed.")
    print("  Your model is working well but may need more training for edge cases.")
elif passed >= total * 0.5:
    print("⚠️  Partial success. Some tests passed.")
    print("   Consider training for more epochs or adjusting hyperparameters.")
else:
    print("❌ Most tests failed.")
    print("   The model may need more training data or different configuration.")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Try generating workflows with your own prompts")
print("2. Test the generated workflows in ComfyUI")
print("3. If results are good, proceed to Phase 7 for detailed evaluation")
print("4. If results need improvement, consider:")
print("   • Training for more epochs")
print("   • Expanding the dataset")
print("   • Adjusting LoRA rank or learning rate")
print("="*70)

# Interactive mode
print("\n" + "="*70)
print("INTERACTIVE MODE")
print("="*70)
print("\nWould you like to try generating a custom workflow?")
print("You can modify this script or create your own prompts.")
print("\nExample usage:")
print("""
instruction = "Generate a ComfyUI workflow for text-to-image generation"
parameters = "Model: sd-v1-5, Prompt: 'your prompt here', Steps: 20, CFG: 7, Size: 512x512"
workflow = generate_workflow(instruction, parameters)
print(workflow)
""")
print("="*70)

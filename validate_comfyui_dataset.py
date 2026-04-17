"""
ComfyUI Dataset Validator
Checks dataset quality and identifies issues
"""

import json
import numpy as np
from collections import Counter

def load_dataset(filename):
    """Load JSONL dataset"""
    dataset = []
    with open(filename, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def validate_json_syntax(dataset):
    """Check if all outputs are valid JSON"""
    print("\n" + "="*70)
    print("JSON SYNTAX VALIDATION")
    print("="*70)
    
    invalid_count = 0
    invalid_examples = []
    
    for i, example in enumerate(dataset):
        try:
            workflow = json.loads(example["output"])
            # Check for required keys
            if "nodes" not in workflow or "links" not in workflow:
                invalid_count += 1
                invalid_examples.append((i, "Missing 'nodes' or 'links' key"))
        except json.JSONDecodeError as e:
            invalid_count += 1
            invalid_examples.append((i, f"JSON decode error: {str(e)}"))
    
    if invalid_count == 0:
        print("✓ All examples have valid JSON syntax")
    else:
        print(f"✗ Found {invalid_count} invalid examples:")
        for idx, error in invalid_examples[:5]:  # Show first 5
            print(f"  Example {idx}: {error}")
    
    return invalid_count == 0

def check_field_completeness(dataset):
    """Check if all required fields are present"""
    print("\n" + "="*70)
    print("FIELD COMPLETENESS CHECK")
    print("="*70)
    
    required_fields = ["instruction", "input", "output"]
    issues = []
    
    for i, example in enumerate(dataset):
        for field in required_fields:
            if field not in example:
                issues.append((i, f"Missing field: {field}"))
            elif not example[field] or not example[field].strip():
                issues.append((i, f"Empty field: {field}"))
    
    if not issues:
        print("✓ All examples have complete fields")
    else:
        print(f"✗ Found {len(issues)} field issues:")
        for idx, issue in issues[:5]:
            print(f"  Example {idx}: {issue}")
    
    return len(issues) == 0

def analyze_diversity(dataset):
    """Analyze dataset diversity"""
    print("\n" + "="*70)
    print("DIVERSITY ANALYSIS")
    print("="*70)
    
    # Instruction diversity
    instructions = [ex["instruction"] for ex in dataset]
    unique_instructions = len(set(instructions))
    print(f"\nInstructions:")
    print(f"  Total: {len(instructions)}")
    print(f"  Unique: {unique_instructions}")
    print(f"  Diversity: {unique_instructions/len(instructions)*100:.1f}%")
    
    # Most common instructions
    instruction_counts = Counter(instructions)
    print(f"\n  Most common:")
    for instruction, count in instruction_counts.most_common(3):
        print(f"    '{instruction[:50]}...': {count} times")
    
    # Output diversity
    outputs = [ex["output"] for ex in dataset]
    unique_outputs = len(set(outputs))
    print(f"\nOutputs:")
    print(f"  Total: {len(outputs)}")
    print(f"  Unique: {unique_outputs}")
    print(f"  Diversity: {unique_outputs/len(outputs)*100:.1f}%")
    
    if unique_outputs < len(outputs) * 0.8:
        print(f"  ⚠️ Warning: Low output diversity ({unique_outputs/len(outputs)*100:.1f}%)")
        print(f"     Consider adding more variations")
    else:
        print(f"  ✓ Good output diversity")

def analyze_lengths(dataset):
    """Analyze text lengths"""
    print("\n" + "="*70)
    print("LENGTH ANALYSIS")
    print("="*70)
    
    # Instruction lengths
    instruction_lengths = [len(ex["instruction"]) for ex in dataset]
    print(f"\nInstruction lengths (characters):")
    print(f"  Mean: {np.mean(instruction_lengths):.0f}")
    print(f"  Min: {np.min(instruction_lengths)}")
    print(f"  Max: {np.max(instruction_lengths)}")
    
    # Input lengths
    input_lengths = [len(ex["input"]) for ex in dataset]
    print(f"\nInput lengths (characters):")
    print(f"  Mean: {np.mean(input_lengths):.0f}")
    print(f"  Min: {np.min(input_lengths)}")
    print(f"  Max: {np.max(input_lengths)}")
    
    # Output lengths
    output_lengths = [len(ex["output"]) for ex in dataset]
    print(f"\nOutput lengths (characters):")
    print(f"  Mean: {np.mean(output_lengths):.0f}")
    print(f"  Min: {np.min(output_lengths)}")
    print(f"  Max: {np.max(output_lengths)}")
    
    # Check for outliers
    output_mean = np.mean(output_lengths)
    output_std = np.std(output_lengths)
    outliers = [i for i, length in enumerate(output_lengths) 
                if abs(length - output_mean) > 3 * output_std]
    
    if outliers:
        print(f"\n  ⚠️ Found {len(outliers)} output length outliers")
        print(f"     Examples: {outliers[:5]}")
    else:
        print(f"\n  ✓ No significant length outliers")

def check_workflow_validity(dataset):
    """Check if workflows have valid structure"""
    print("\n" + "="*70)
    print("WORKFLOW STRUCTURE VALIDATION")
    print("="*70)
    
    issues = []
    
    for i, example in enumerate(dataset):
        try:
            workflow = json.loads(example["output"])
            
            # Check nodes
            if "nodes" not in workflow or not workflow["nodes"]:
                issues.append((i, "No nodes in workflow"))
                continue
            
            # Check node structure
            for node_idx, node in enumerate(workflow["nodes"]):
                if "id" not in node:
                    issues.append((i, f"Node {node_idx} missing 'id'"))
                if "type" not in node:
                    issues.append((i, f"Node {node_idx} missing 'type'"))
            
            # Check links
            if "links" not in workflow:
                issues.append((i, "No links in workflow"))
            elif workflow["links"] and len(workflow["links"]) > 0:
                # Validate link structure
                for link_idx, link in enumerate(workflow["links"]):
                    if not isinstance(link, list) or len(link) < 5:
                        issues.append((i, f"Link {link_idx} has invalid structure"))
        
        except Exception as e:
            issues.append((i, f"Error parsing workflow: {str(e)}"))
    
    if not issues:
        print("✓ All workflows have valid structure")
    else:
        print(f"✗ Found {len(issues)} workflow issues:")
        for idx, issue in issues[:10]:
            print(f"  Example {idx}: {issue}")
    
    return len(issues) == 0

def generate_report(dataset, filename):
    """Generate a summary report"""
    print("\n" + "="*70)
    print("DATASET SUMMARY REPORT")
    print("="*70)
    
    print(f"\nDataset: {filename}")
    print(f"Total examples: {len(dataset)}")
    
    # Count workflow types
    instructions = [ex["instruction"] for ex in dataset]
    instruction_counts = Counter(instructions)
    
    print(f"\nWorkflow types:")
    for instruction, count in instruction_counts.most_common():
        percentage = count / len(dataset) * 100
        print(f"  {instruction}: {count} ({percentage:.1f}%)")
    
    # Quality score
    print(f"\nQuality indicators:")
    unique_outputs = len(set(ex["output"] for ex in dataset))
    diversity_score = unique_outputs / len(dataset) * 100
    print(f"  Output diversity: {diversity_score:.1f}%")
    
    avg_output_length = np.mean([len(ex["output"]) for ex in dataset])
    print(f"  Avg output length: {avg_output_length:.0f} characters")
    
    # Recommendations
    print(f"\nRecommendations:")
    if len(dataset) < 500:
        print(f"  • Consider expanding to 500+ examples for better quality")
    if diversity_score < 80:
        print(f"  • Add more variations to improve diversity")
    if avg_output_length < 500:
        print(f"  • Outputs seem short - verify workflow completeness")
    
    print(f"\n✓ Validation complete!")

if __name__ == "__main__":
    import sys
    
    filename = "comfyui_dataset.jsonl"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    print("="*70)
    print("COMFYUI DATASET VALIDATOR")
    print("="*70)
    print(f"\nValidating: {filename}")
    
    try:
        # Load dataset
        dataset = load_dataset(filename)
        print(f"✓ Loaded {len(dataset)} examples")
        
        # Run validations
        field_ok = check_field_completeness(dataset)
        json_ok = validate_json_syntax(dataset)
        workflow_ok = check_workflow_validity(dataset)
        
        # Analyze
        analyze_diversity(dataset)
        analyze_lengths(dataset)
        
        # Report
        generate_report(dataset, filename)
        
        # Final verdict
        print("\n" + "="*70)
        if field_ok and json_ok and workflow_ok:
            print("✓ DATASET IS VALID AND READY FOR TRAINING")
        else:
            print("✗ DATASET HAS ISSUES - PLEASE FIX BEFORE TRAINING")
        print("="*70)
        
    except FileNotFoundError:
        print(f"\n✗ Error: File '{filename}' not found")
        print("Run create_comfyui_dataset.py first to generate a dataset")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")

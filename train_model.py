"""
Phase 6: First Training Run
Train TinyLlama on ComfyUI workflow generation dataset
"""

# CRITICAL: Import CPU optimization FIRST before any other imports
import cpu_setup

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import time
import os

print("="*70)
print("PHASE 6: FIRST TRAINING RUN")
print("="*70)
print("\nThis script will train TinyLlama on your ComfyUI workflow dataset.")
print("Expected training time: ~50 minutes for 500 examples × 3 epochs")
print("="*70)

# Step 1: Load Model
print("\n[1/7] Loading TinyLlama model...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Added padding token")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

print(f"✓ Model loaded")
print(f"  Parameters: {model.num_parameters():,}")
print(f"  Memory: {model.get_memory_footprint() / 1e9:.2f} GB")

# Step 2: Apply LoRA
print("\n[2/7] Applying LoRA configuration...")
lora_config = LoraConfig(
    r=8,                              # Rank (balance between speed and quality)
    lora_alpha=16,                    # Scaling factor (2 × r)
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,                # Regularization
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# FIX 1: Required when using gradient_checkpointing with LoRA/PEFT.
# Gradient checkpointing discards intermediate activations to save memory, then
# recomputes them on the backward pass. With LoRA, the frozen base model layers
# have requires_grad=False, which breaks the autograd graph at the input
# embedding stage — causing the "element 0 of tensors does not require grad"
# error. This call registers a forward hook that forces input embeddings to
# retain their grad_fn so the gradient chain stays intact through checkpointing.
model.enable_input_require_grads()

print("✓ LoRA applied")
model.print_trainable_parameters()

# Step 3: Load Dataset
print("\n[3/7] Loading formatted dataset...")
dataset = load_dataset("json", data_files="comfyui_dataset_formatted.jsonl", split="train")
print(f"✓ Loaded {len(dataset)} examples")
print(f"  Columns: {dataset.column_names}")

# Preview first example
print("\n  Preview of first example:")
print(f"  Text length: {len(dataset[0]['text'])} characters")
print(f"  First 150 chars: {dataset[0]['text'][:150]}...")

# Step 4: Configure Training
print("\n[4/7] Configuring training parameters...")

training_args = TrainingArguments(
    # Output
    output_dir="./training_output",
    run_name="comfyui-tinyllama-lora",
    
    # Training duration
    num_train_epochs=3,                    # 3 passes through the data
    
    # Batch size and accumulation
    per_device_train_batch_size=1,         # Process 1 example at a time
    gradient_accumulation_steps=8,         # Accumulate 8 steps = effective batch of 8
    
    # Learning rate
    learning_rate=2e-4,                    # Standard for LoRA fine-tuning
    lr_scheduler_type="cosine",            # Gradually decrease learning rate
    warmup_steps=10,                       # Warm up for first 10 steps
    
    # Optimization
    optim="adamw_torch",                   # Adam optimizer
    gradient_checkpointing=True,           # Save memory by recomputing gradients
    # FIX 2: Suppress the use_reentrant deprecation warning from PyTorch 2.9+.
    # use_reentrant=False is the modern recommended mode for gradient checkpointing.
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Logging and saving
    logging_steps=5,                       # Log every 5 steps
    save_steps=50,                         # Save checkpoint every 50 steps
    save_total_limit=2,                    # Keep only 2 most recent checkpoints
    
    # Evaluation
    evaluation_strategy="no",              # No validation set for now
    
    # Other
    report_to="none",                      # Don't send to wandb/tensorboard
    seed=42,                               # Reproducibility
    fp16=False,                            # CPU doesn't support fp16
    dataloader_num_workers=0,              # Single worker for CPU
)

print("✓ Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Total steps: ~{len(dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

# Step 5: Create Trainer
print("\n[5/7] Creating SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,                   # Handle long ComfyUI workflows
    packing=False,                         # Don't pack multiple examples together
)

print("✓ Trainer created")
print(f"  Max sequence length: 1024 tokens")
print(f"  Dataset text field: 'text'")

# Step 6: Train
print("\n[6/7] Starting training...")
print("="*70)
print("TRAINING IN PROGRESS")
print("="*70)
print("\nWhat to watch for:")
print("  • Loss should decrease from ~2.5 to ~0.8-1.5")
print("  • Training speed: ~2 seconds per step")
print("  • Memory usage: Should stay under 8 GB")
print("\nYou can press Ctrl+C to stop training early (model will be saved)")
print("="*70)
print()

start_time = time.time()

try:
    # Train the model
    trainer.train()
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Training time: {training_time:.2f} minutes")
    print(f"Average time per epoch: {training_time / training_args.num_train_epochs:.2f} minutes")
    
except KeyboardInterrupt:
    print("\n" + "="*70)
    print("⚠️  TRAINING INTERRUPTED BY USER")
    print("="*70)
    print("Saving current progress...")
    trainer.save_model("./training_interrupted")
    print("✓ Model saved to: ./training_interrupted")
    print("\nYou can resume training later or use this checkpoint for testing.")
    exit(0)
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR DURING TRAINING")
    print("="*70)
    print(f"Error: {e}")
    print("\nSaving checkpoint before exit...")
    trainer.save_model("./training_error_checkpoint")
    print("✓ Checkpoint saved to: ./training_error_checkpoint")
    raise

# Step 7: Save Model
print("\n[7/7] Saving fine-tuned model...")

# Save LoRA adapters
output_dir = "./comfyui_lora_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✓ Model saved to: {output_dir}")

# Get final model size
adapter_size = sum(os.path.getsize(os.path.join(output_dir, f)) 
                   for f in os.listdir(output_dir) 
                   if os.path.isfile(os.path.join(output_dir, f)))
print(f"  LoRA adapter size: {adapter_size / 1e6:.2f} MB")

# Training summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"✓ Training completed in {training_time:.2f} minutes")
print(f"✓ Model saved to: {output_dir}")
print(f"✓ Checkpoints saved to: ./training_output")
print("\nModel details:")
print(f"  • Base model: {model_name}")
print(f"  • LoRA rank: {lora_config.r}")
print(f"  • Target modules: {lora_config.target_modules}")
print(f"  • Training examples: {len(dataset)}")
print(f"  • Epochs: {training_args.num_train_epochs}")
print(f"  • Total training steps: {trainer.state.global_step}")

print("\n" + "="*70)
print("NEXT STEPS - PHASE 7")
print("="*70)
print("1. Test the fine-tuned model with test_model.py")
print("2. Compare outputs with the base model")
print("3. Evaluate on different workflow types")
print("4. Check if the model generates valid ComfyUI JSON")
print("\nTo test your model, run:")
print("  python test_model.py")
print("="*70)

print("\n✓ Phase 6 complete! Your model is ready for testing.")
"""
Training Script Optimized for AMD EPYC 7402P Server
48 cores, optimized for maximum throughput
"""

# CRITICAL: Import CPU optimization FIRST
import cpu_setup_server

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import time
import os

print("="*70)
print("COMFYUI TRAINING - SERVER OPTIMIZED")
print("="*70)
print("\nServer: AMD EPYC 7402P 24-Core (48 threads)")
print("Dataset: Combined synthetic + real workflows")
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
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

print("✓ LoRA applied")
model.print_trainable_parameters()

# Step 3: Load Dataset
print("\n[3/7] Loading formatted dataset...")
dataset = load_dataset("json", data_files="comfyui_dataset_combined_formatted.jsonl", split="train")
print(f"✓ Loaded {len(dataset)} examples")
print(f"  Columns: {dataset.column_names}")

# Preview
print(f"\n  First example preview:")
print(f"  Text length: {len(dataset[0]['text'])} characters")

# Step 4: Configure Training - OPTIMIZED FOR 48 CORES
print("\n[4/7] Configuring training (48-core optimization)...")

training_args = TrainingArguments(
    # Output
    output_dir="./training_output_server",
    run_name="comfyui-tinyllama-server",
    
    # Training duration
    num_train_epochs=3,
    
    # Batch size - OPTIMIZED FOR SERVER
    per_device_train_batch_size=2,         # 2x larger than laptop (more RAM)
    gradient_accumulation_steps=4,         # Effective batch = 8 (same as before)
    
    # Learning rate
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=20,                       # More warmup for larger dataset
    
    # Optimization
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Parallelization - OPTIMIZED FOR 48 CORES
    dataloader_num_workers=8,              # Use 8 workers for data loading
    dataloader_pin_memory=False,           # CPU training doesn't need pinned memory
    
    # Logging and saving
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    
    # Evaluation
    evaluation_strategy="no",
    
    # Other
    report_to="none",
    seed=42,
    fp16=False,
    
    # Performance
    max_grad_norm=1.0,                     # Gradient clipping for stability
)

print("✓ Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Data workers: {training_args.dataloader_num_workers}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Total steps: ~{len(dataset) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

# Estimate training time
examples_per_hour = 250  # Conservative estimate for 48-core server
total_examples = len(dataset) * training_args.num_train_epochs
estimated_hours = total_examples / examples_per_hour

print(f"\n  Estimated training time: {estimated_hours:.1f} hours")

# Step 5: Create Trainer
print("\n[5/7] Creating SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,                   # Adjust based on format_combined_dataset.py output
    packing=False,
)

print("✓ Trainer created")

# Step 6: Train
print("\n[6/7] Starting training...")
print("="*70)
print("TRAINING IN PROGRESS")
print("="*70)
print("\nMonitor:")
print("  • Loss should decrease from ~2.0 to ~0.5-0.8")
print("  • Grad norm should be 0.5-2.0 (NOT 0.0!)")
print("  • Training speed: ~1-2 seconds per step on 48 cores")
print("\nPress Ctrl+C to stop (model will be saved)")
print("="*70)
print()

start_time = time.time()

try:
    trainer.train()
    
    end_time = time.time()
    training_time = (end_time - start_time) / 3600  # hours
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Training time: {training_time:.2f} hours")
    print(f"Average time per epoch: {training_time / training_args.num_train_epochs:.2f} hours")
    
except KeyboardInterrupt:
    print("\n" + "="*70)
    print("⚠️  TRAINING INTERRUPTED BY USER")
    print("="*70)
    print("Saving current progress...")
    trainer.save_model("./training_interrupted_server")
    print("✓ Model saved to: ./training_interrupted_server")
    exit(0)
    
except Exception as e:
    print("\n" + "="*70)
    print("❌ ERROR DURING TRAINING")
    print("="*70)
    print(f"Error: {e}")
    print("\nSaving checkpoint...")
    trainer.save_model("./training_error_checkpoint_server")
    print("✓ Checkpoint saved to: ./training_error_checkpoint_server")
    raise

# Step 7: Save Model
print("\n[7/7] Saving fine-tuned model...")

output_dir = "./comfyui_lora_model_final"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✓ Model saved to: {output_dir}")

# Get model size
adapter_size = sum(os.path.getsize(os.path.join(output_dir, f)) 
                   for f in os.listdir(output_dir) 
                   if os.path.isfile(os.path.join(output_dir, f)))
print(f"  LoRA adapter size: {adapter_size / 1e6:.2f} MB")

# Training summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"✓ Training completed in {training_time:.2f} hours")
print(f"✓ Model saved to: {output_dir}")
print(f"✓ Checkpoints saved to: ./training_output_server")
print("\nModel details:")
print(f"  • Base model: {model_name}")
print(f"  • LoRA rank: {lora_config.r}")
print(f"  • Target modules: {lora_config.target_modules}")
print(f"  • Training examples: {len(dataset)}")
print(f"  • Epochs: {training_args.num_train_epochs}")
print(f"  • Total training steps: {trainer.state.global_step}")
print(f"  • Final loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("1. Check your email for completion notification")
print("2. Test the model: python test_model.py")
print("3. Download the model from server if needed")
print("="*70)

# Send email notification
print("\n" + "="*70)
print("SENDING EMAIL NOTIFICATION")
print("="*70)

# Get recipient email from environment variable
recipient_email = os.environ.get("TRAINING_NOTIFICATION_EMAIL", None)

if recipient_email and recipient_email != "user@example.com":
    try:
        # Import email function
        from send_email_notification import send_training_complete_email
        
        send_training_complete_email(
            recipient_email=recipient_email,
            training_time_hours=training_time,
            final_loss=trainer.state.log_history[-1].get('loss', None),
            total_steps=trainer.state.global_step,
            model_path=output_dir
        )
    except Exception as e:
        print(f"⚠️  Email notification failed (non-critical): {e}")
        print("   Training completed successfully, but email could not be sent.")
        print("   You can check the model manually.")
else:
    print("⚠️  Email notification skipped (TRAINING_NOTIFICATION_EMAIL not set)")
    print("   Training completed successfully!")

print("="*70)

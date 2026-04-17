"""
Resume Training from Checkpoint
Continue training from where you stopped
"""

# CRITICAL: Import CPU optimization FIRST
import cpu_setup

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import time
import os

print("="*70)
print("RESUMING TRAINING FROM CHECKPOINT")
print("="*70)

# Load base model
print("\n[1/5] Loading base model...")
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
print("✓ Base model loaded")

# Load LoRA adapters from interrupted checkpoint
print("\n[2/5] Loading LoRA adapters from checkpoint...")
model = PeftModel.from_pretrained(base_model, "./training_interrupted")
model.enable_input_require_grads()
print("✓ LoRA adapters loaded from interrupted training")

# Load dataset
print("\n[3/5] Loading dataset...")
dataset = load_dataset("json", data_files="comfyui_dataset_formatted.jsonl", split="train")
print(f"✓ Loaded {len(dataset)} examples")

# Configure training - CONTINUE from where we stopped
print("\n[4/5] Configuring training to continue...")

training_args = TrainingArguments(
    output_dir="./training_output",
    run_name="comfyui-tinyllama-lora-resumed",
    
    # Training duration - complete at least 1 full epoch
    num_train_epochs=1.5,  # Will train from 0.8 to 1.5 (0.7 more epochs)
    
    # Same settings as before
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=5,  # Fewer warmup steps since we're resuming
    
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    logging_steps=5,
    save_steps=25,  # Save more frequently
    save_total_limit=3,
    
    evaluation_strategy="no",
    report_to="none",
    seed=42,
    fp16=False,
    dataloader_num_workers=0,
)

print("✓ Training will continue for 0.7 more epochs")
print(f"  Target: Complete at least epoch 1.0")
print(f"  Estimated time: ~45 minutes")

# Create trainer
print("\n[5/5] Creating trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=False,
)

print("✓ Trainer ready")

# Resume training
print("\n" + "="*70)
print("RESUMING TRAINING")
print("="*70)
print("\nCurrent status: Epoch 0.8, Loss 0.51")
print("Target: Epoch 1.5, Loss ~0.3-0.4")
print("\nWhat to expect:")
print("  • Loss should continue decreasing to ~0.3-0.4")
print("  • Model will learn correct node types")
print("  • JSON generation will become consistent")
print("="*70)
print()

start_time = time.time()

try:
    trainer.train()
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED!")
    print("="*70)
    print(f"Additional training time: {training_time:.2f} minutes")
    
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted again")
    trainer.save_model("./training_interrupted_2")
    print("✓ Saved to: ./training_interrupted_2")
    exit(0)

# Save final model
print("\nSaving final model...")
output_dir = "./comfyui_lora_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ Final model saved to: {output_dir}")

print("\n" + "="*70)
print("TRAINING COMPLETE - READY FOR TESTING")
print("="*70)
print("\nRun: python test_model.py")
print("="*70)

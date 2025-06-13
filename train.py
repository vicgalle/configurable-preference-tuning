#!/usr/bin/env python3
"""
Fine-tuning script

This script fine-tunes a language model on the creative-rubrics-preferences dataset,
for the "Configurable Preference Tuning with Rubric-Guided Synthetic Data" paper.

Requirements:
    - unsloth
    - transformers
    - trl
    - datasets
    - torch

Usage:
    python train.py
"""

import os
from datasets import load_dataset
from transformers import TrainingArguments
from trl import DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

# Configuration
CONFIG = {
    "model_name": "mistralai/Mistral-Nemo-Instruct-2407",
    "max_seq_length": 2048,
    "load_in_4bit": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "num_train_epochs": 1,
    "beta": 0.1,
    "max_prompt_length": 256,
    "output_dir": "outputs",
    "checkpoint_dir": "checkpoints/creative_dpo_model",
    "random_state": 42,
}


def load_and_prepare_dataset():
    """Load and prepare the creative rubrics preferences dataset."""
    print("Loading dataset...")
    dataset = load_dataset('vicgalle/creative-rubrics-preferences')['train']
    
    print(f"Dataset loaded with {len(dataset)} examples")
    return dataset


def setup_model_and_tokenizer():
    """Initialize and configure the model and tokenizer."""
    print("Setting up model and tokenizer...")
    
    # Apply DPO patches
    PatchDPOTrainer()
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=CONFIG["load_in_4bit"],
    )
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["random_state"],
        max_seq_length=CONFIG["max_seq_length"],
    )
    
    print("Model and tokenizer setup complete")
    return model, tokenizer


def create_training_arguments():
    """Create training arguments for DPO training."""
    return TrainingArguments(
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        warmup_ratio=CONFIG["warmup_ratio"],
        num_train_epochs=CONFIG["num_train_epochs"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        seed=CONFIG["random_state"],
        output_dir=CONFIG["output_dir"],
        report_to=None,  # Disable wandb/tensorboard logging by default
        save_strategy="epoch",
        save_total_limit=2,
    )


def main():
    """Main training function."""
    print("Starting DPO fine-tuning...")
    
    # Load dataset
    dataset = load_and_prepare_dataset()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Create training arguments
    training_args = create_training_arguments()
    
    # Initialize DPO trainer
    print("Initializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Use implicit reference model to save VRAM
        args=training_args,
        beta=CONFIG["beta"],
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=CONFIG["max_seq_length"],
        max_prompt_length=CONFIG["max_prompt_length"],
    )
    
    # Start training
    print("Starting training...")
    dpo_trainer.train()
    
    # Save the final model
    print(f"Saving model to {CONFIG['checkpoint_dir']}...")
    os.makedirs(os.path.dirname(CONFIG["checkpoint_dir"]), exist_ok=True)
    dpo_trainer.save_model(CONFIG["checkpoint_dir"])
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
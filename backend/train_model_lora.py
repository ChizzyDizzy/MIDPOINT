"""
SafeMind AI - LoRA Fine-Tuning Script
Efficient fine-tuning using LoRA (Low-Rank Adaptation)

This script implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA
for training a mental health chatbot on consumer hardware.
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import os
import argparse
from datetime import datetime

def load_dataset(dataset_path: str):
    """Load and prepare training dataset"""
    print(f"\n[1/8] Loading dataset from {dataset_path}")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract samples
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("Invalid dataset format")

    print(f"✓ Loaded {len(samples)} samples")

    # Format for training (instruction format)
    formatted_samples = []
    for sample in samples:
        text = f"""### Instruction:
{sample.get('instruction', 'You are a mental health awareness chatbot.')}

### Input:
{sample.get('input', '')}

### Response:
{sample.get('response', '')}"""
        formatted_samples.append({"text": text})

    return formatted_samples

def load_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    """Load base model and tokenizer with optional quantization"""
    print(f"\n[2/8] Loading model: {model_name}")

    # Quantization config for 4-bit training (saves memory)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization (QLoRA)")
    else:
        bnb_config = None

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not use_4bit else None
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("✓ Model and tokenizer loaded")

    return model, tokenizer

def configure_lora(model, lora_r: int = 8, lora_alpha: int = 16,
                  lora_dropout: float = 0.1):
    """Configure LoRA adapters"""
    print(f"\n[3/8] Configuring LoRA (r={lora_r}, alpha={lora_alpha})")

    # Prepare model for k-bit training if quantized
    if hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,  # Rank (higher = more parameters, better quality, slower)
        lora_alpha=lora_alpha,  # Scaling factor
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Which layers to adapt
        # For different models, target_modules may vary:
        # Phi-3: ["q_proj", "k_proj", "v_proj", "o_proj"]
        # LLaMA: ["q_proj", "v_proj"]
        # GPT: ["c_attn"]
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA configured")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Total parameters: {total_params:,}")

    return model

def prepare_dataset(samples, tokenizer, max_length: int = 512):
    """Tokenize dataset"""
    print(f"\n[4/8] Preparing dataset (max_length={max_length})")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )

    # Create HuggingFace dataset
    dataset = Dataset.from_list(samples)

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=['text'],
        batched=True,
        desc="Tokenizing"
    )

    print(f"✓ Prepared {len(tokenized_dataset)} tokenized samples")

    return tokenized_dataset

def train_model(model, tokenizer, dataset, output_dir: str,
                num_epochs: int = 3, batch_size: int = 4,
                learning_rate: float = 2e-4):
    """Train the model with LoRA"""
    print(f"\n[5/8] Setting up training")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        learning_rate=learning_rate,
        fp16=True,  # Mixed precision training
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        logging_dir=os.path.join(output_dir, 'logs'),
        report_to="none",  # Disable wandb
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        # Memory optimizations
        gradient_checkpointing=True,
        max_grad_norm=0.3,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("✓ Training configuration ready")
    print("\n[6/8] Starting training...")
    print("=" * 60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size} (effective: {batch_size * 4})")
    print(f"Learning rate: {learning_rate}")
    print(f"Expected time: 30-60 minutes (depends on GPU)")
    print("=" * 60)
    print()

    # Train
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    training_time = (end_time - start_time).total_seconds()

    print("\n" + "=" * 60)
    print(f"✓ Training complete!")
    print(f"  Total time: {training_time/60:.1f} minutes")
    print("=" * 60)

    return trainer

def save_model(model, tokenizer, output_dir: str):
    """Save LoRA adapters and tokenizer"""
    print(f"\n[7/8] Saving model to {output_dir}")

    # Save LoRA adapters
    model.save_pretrained(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info = {
        "model_type": "LoRA adapter",
        "base_model": model.config._name_or_path,
        "trained_at": datetime.now().isoformat(),
        "framework": "PEFT + LoRA"
    }

    with open(os.path.join(output_dir, "training_info.json"), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"✓ Model saved")
    print(f"  LoRA adapters: {output_dir}/adapter_model.bin")
    print(f"  Config: {output_dir}/adapter_config.json")
    print(f"  Tokenizer: {output_dir}/tokenizer_config.json")

def test_model(model, tokenizer):
    """Quick test of trained model"""
    print("\n[8/8] Testing trained model...")

    test_prompts = [
        "I feel very anxious about my exams",
        "My family doesn't understand me",
        "I feel hopeless",
    ]

    for prompt in test_prompts:
        full_prompt = f"""### Instruction:
You are a mental health awareness chatbot. Provide empathetic support.

### Input:
{prompt}

### Response:
"""

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:")[-1].strip()

        print(f"\nTest: {prompt}")
        print(f"Response: {response[:200]}...")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Train SafeMind AI with LoRA")
    parser.add_argument("--dataset", default="../data/synthetic_training_data.json",
                       help="Path to training dataset")
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct",
                       help="Base model name")
    parser.add_argument("--output", default="./safemind-lora-model",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--no-4bit", action="store_true",
                       help="Disable 4-bit quantization")
    parser.add_argument("--skip-test", action="store_true",
                       help="Skip testing after training")

    args = parser.parse_args()

    print("=" * 60)
    print("SafeMind AI - LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Base model: {args.model}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Load dataset
    samples = load_dataset(args.dataset)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, use_4bit=not args.no_4bit)

    # Configure LoRA
    model = configure_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    # Prepare dataset
    dataset = prepare_dataset(samples, tokenizer, max_length=args.max_length)

    # Train
    trainer = train_model(
        model, tokenizer, dataset, args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    # Save
    save_model(model, tokenizer, args.output)

    # Test
    if not args.skip_test:
        test_model(model, tokenizer)

    print("\n✅ Complete! Next steps:")
    print(f"1. Update .env: AI_BACKEND=local")
    print(f"2. Update .env: LOCAL_MODEL={args.output}")
    print("3. Update ai_model_free.py to support LoRA loading")
    print("4. Run: python app_improved.py")
    print("5. Test: python test_mvp.py")

if __name__ == "__main__":
    main()

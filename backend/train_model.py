"""
SafeMind AI - Model Training Script
Fine-tunes DialoGPT on the mental health dataset for empathetic responses.
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

print("=" * 70)
print(" " * 15 + "SafeMind AI - Model Training")
print("=" * 70)
print()

# Configuration
BASE_MODEL = "microsoft/DialoGPT-small"  # Change to DialoGPT-medium for better quality
OUTPUT_DIR = "./safemind-mental-health-model"
DATASET_PATH = "../data/mental_health_dataset.json"
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

# Step 1: Load dataset
print("[1/7] Loading dataset...")
print("-" * 70)

try:
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Dataset not found at {DATASET_PATH}")
    print("Run this first:")
    print("  cd ../scripts")
    print("  python expand_dataset.py --num-samples 4000 --output ../data/mental_health_dataset.json")
    exit(1)

samples = data.get('samples', [])
if not samples:
    # Fallback: try old format
    conversations = data.get('conversations', [])
    if conversations:
        samples = []
        for conv in conversations:
            samples.append({
                'input': conv.get('user_input', ''),
                'response': conv.get('expected_response', ''),
                'category': conv.get('category', 'general'),
                'risk_level': conv.get('risk_level', 'low')
            })

print(f"Loaded {len(samples)} training samples")
categories = set(s.get('category', 'unknown') for s in samples)
print(f"  Categories: {categories}")
print()

# Step 2: Prepare training data
print("[2/7] Preparing training data...")
print("-" * 70)

training_texts = []
for sample in samples:
    user_input = sample.get('input', '')
    response = sample.get('response', '')
    if user_input and response:
        text = f"<|user|> {user_input}\n<|assistant|> {response}<|endoftext|>"
        training_texts.append(text)

print(f"Prepared {len(training_texts)} training examples")
print(f"  Average length: {sum(len(t) for t in training_texts) // max(len(training_texts), 1)} characters")
print()

# Step 3: Load base model
print("[3/7] Loading base model...")
print("-" * 70)
print(f"Model: {BASE_MODEL}")
print()

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Base model loaded successfully")
    print(f"  Model parameters: {model.num_parameters():,}")
    print()

except Exception as e:
    print(f"ERROR loading model: {e}")
    print("Install dependencies: pip install transformers torch")
    exit(1)

# Step 4: Tokenize dataset
print("[4/7] Tokenizing dataset...")
print("-" * 70)

train_dataset = Dataset.from_dict({'text': training_texts})

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
    )

tokenized_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=['text'],
    batched=True,
    desc="Tokenizing"
)

print(f"Tokenized {len(tokenized_dataset)} examples")
print(f"  Max sequence length: {MAX_LENGTH} tokens")
print()

# Step 5: Set up training configuration
print("[5/7] Setting up training configuration...")
print("-" * 70)

use_fp16 = False

if torch.cuda.is_available():
    device = "cuda"
    print(f"Training device: NVIDIA GPU")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    use_fp16 = True
    print(f"  FP16 mixed precision: enabled")
else:
    device = "cpu"
    # Check for AMD GPU (not supported by PyTorch CUDA)
    try:
        import subprocess
        gpu_check = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
            capture_output=True, text=True, timeout=5
        )
        gpu_name = gpu_check.stdout.strip()
        if any(keyword in gpu_name.lower() for keyword in ['amd', 'radeon', 'rx ']):
            print("Training device: CPU")
            print()
            print("  NOTE: AMD GPU detected but PyTorch CUDA only supports NVIDIA GPUs.")
            print("  Your AMD GPU cannot be used for training with PyTorch on Windows.")
            print()
            print("  RECOMMENDED: Use Google Colab for free NVIDIA T4 GPU training.")
            print("  See GUIDE_CLOUD_TRAINING.md for step-by-step instructions.")
            print("  Training on Colab takes ~15-20 minutes vs hours on CPU.")
        else:
            print("Training device: CPU")
            print("  For faster training, see GUIDE_CLOUD_TRAINING.md")
    except Exception:
        print("Training device: CPU")
        print("  For faster training, see GUIDE_CLOUD_TRAINING.md")

    # Optimize for CPU training
    BATCH_SIZE = 2
    NUM_EPOCHS = 3
    print()
    print(f"  CPU mode: batch size reduced to {BATCH_SIZE}, epochs to {NUM_EPOCHS}")

print()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    save_total_limit=2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=25,
    logging_dir='./logs',
    report_to="none",
    fp16=use_fp16,
    dataloader_pin_memory=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print(f"Training configuration ready")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  FP16: {use_fp16}")
print()

# Step 6: Train
print("[6/7] Training model...")
print("=" * 70)
print()
print("Watch the 'loss' value - it should decrease over time.")
print("Lower loss = better model performance")
print()

try:
    trainer.train()
    print()
    print("=" * 70)
    print("Training completed successfully!")
    print("=" * 70)
    print()

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    print("Partial progress has been saved")
except Exception as e:
    print(f"\nERROR during training: {e}")
    print("Out of memory? Reduce BATCH_SIZE to 2 or 1")
    exit(1)

# Step 7: Save
print("[7/7] Saving fine-tuned model...")
print("-" * 70)

try:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    training_info = {
        "base_model": BASE_MODEL,
        "training_samples": len(training_texts),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "categories": list(categories),
        "model_path": OUTPUT_DIR
    }

    with open(f"{OUTPUT_DIR}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"Model saved to: {OUTPUT_DIR}")
    print()

except Exception as e:
    print(f"ERROR saving model: {e}")
    exit(1)

# Done
print("=" * 70)
print("  Training Complete")
print("=" * 70)
print()
print(f"  Base model: {BASE_MODEL}")
print(f"  Training samples: {len(training_texts)}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Saved to: {OUTPUT_DIR}")
print()
print("Next steps:")
print("  1. Update .env:")
print(f"     HUGGINGFACE_MODEL={OUTPUT_DIR}")
print("     AI_BACKEND=local")
print()
print("  2. Start the app:")
print("     python app_improved.py")
print()
print("  3. Run tests:")
print("     python test_mvp.py")
print()
print("=" * 70)

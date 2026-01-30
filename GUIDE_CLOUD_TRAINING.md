# SafeMind AI - Cloud Training Guide (Google Colab)

**Train your model for FREE on Google Colab with GPU acceleration.**

This guide walks you through training the SafeMind AI model entirely in the cloud using Google Colab's free GPU. This is the recommended approach if your local machine lacks an NVIDIA GPU.

---

## Why Cloud Training?

- **Free GPU** - Google Colab provides a free T4 GPU
- **Faster** - Training takes ~15-20 minutes vs hours on CPU
- **No local setup** - No need for PyTorch/CUDA on your machine
- **4000 samples** - Train with a larger dataset for better accuracy

---

## Step 1: Generate the Dataset Locally

Before opening Colab, generate the dataset on your machine.

**macOS:**
```bash
cd ~/Documents/MIDPOINT/scripts
source ../backend/venv/bin/activate
python3 expand_dataset.py --num-samples 4000 --output ../data/mental_health_dataset.json
```

**Windows (Git Bash):**
```bash
cd ~/Documents/MIDPOINT/scripts
source ../backend/venv/Scripts/activate
python expand_dataset.py --num-samples 4000 --output ../data/mental_health_dataset.json
```

This creates `data/mental_health_dataset.json` with 4000 training samples.

---

## Step 2: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com)
2. Sign in with your Google account
3. Click **New Notebook**
4. Go to **Runtime > Change runtime type > T4 GPU** then click **Save**

---

## Step 3: Upload, Train, Test, and Download

Copy and paste the **entire script below** into a single Colab cell and run it. It will:

1. Ask you to upload your dataset file
2. Install dependencies
3. Tokenize the dataset
4. Train the model on GPU (~15-20 minutes)
5. Test the model with sample messages
6. Save and download the trained model as a zip file

```python
# ============================================================
# SafeMind AI - Complete Cloud Training Script
# Run this entire cell in Google Colab (Runtime > T4 GPU)
# ============================================================

# --- CONFIGURATION ---
MODEL_NAME = "microsoft/DialoGPT-medium"    # Base model
NUM_EPOCHS = 5                               # Training epochs
BATCH_SIZE = 8                               # Batch size (GPU)
LEARNING_RATE = 2e-5                         # Learning rate
MAX_LENGTH = 512                             # Max token length
WARMUP_STEPS = 100                           # Warmup steps
OUTPUT_DIR = "./safemind-trained-model"       # Output directory

# ============================================================
# STEP 1: Upload dataset
# ============================================================
print("=" * 60)
print("STEP 1: Upload your dataset")
print("=" * 60)
print("Select your mental_health_dataset.json file when prompted.\n")

from google.colab import files
uploaded = files.upload()

# Find the uploaded JSON file
import os
dataset_file = None
for filename in uploaded.keys():
    if filename.endswith('.json'):
        dataset_file = filename
        break

if dataset_file is None:
    raise FileNotFoundError("No JSON file uploaded. Please upload mental_health_dataset.json")

print(f"\nUploaded: {dataset_file}")

# ============================================================
# STEP 2: Install dependencies
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Installing dependencies...")
print("=" * 60 + "\n")

import subprocess
subprocess.check_call([
    "pip", "install", "-q",
    "transformers", "datasets", "accelerate", "torch"
])

print("Dependencies installed.")

# ============================================================
# STEP 3: Load and prepare dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Loading and preparing dataset...")
print("=" * 60 + "\n")

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load dataset
with open(dataset_file, 'r') as f:
    data = json.load(f)

# Support both formats
samples = data.get('samples', [])
if not samples:
    conversations = data.get('conversations', [])
    for conv in conversations:
        samples.append({
            'input': conv.get('user_input', ''),
            'response': conv.get('expected_response', ''),
        })

print(f"Loaded {len(samples)} training samples")

# Format as training text
training_texts = []
for sample in samples:
    user_input = sample.get('input', '')
    response = sample.get('response', '')
    if user_input and response:
        text = f"<|user|> {user_input}\n<|assistant|> {response}<|endoftext|>"
        training_texts.append(text)

print(f"Prepared {len(training_texts)} training examples")
print(f"Average length: {sum(len(t) for t in training_texts) // max(len(training_texts), 1)} characters")

# ============================================================
# STEP 4: Load model and tokenizer
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Loading base model...")
print("=" * 60 + "\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print(f"Model: {MODEL_NAME}")
print(f"Parameters: {model.num_parameters():,}")

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected. Go to Runtime > Change runtime type > T4 GPU")

# ============================================================
# STEP 5: Tokenize dataset
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Tokenizing dataset...")
print("=" * 60 + "\n")

dataset = Dataset.from_dict({"text": training_texts})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing"
)

print(f"Tokenized {len(tokenized_dataset)} examples")
print(f"Max sequence length: {MAX_LENGTH} tokens")

# ============================================================
# STEP 6: Train the model
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Training model...")
print("=" * 60)
print()
print("Watch the 'loss' value - it should decrease over time.")
print("Lower loss = better model performance.")
print()

use_fp16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir="./training-checkpoints",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=WARMUP_STEPS,
    save_steps=500,
    save_total_limit=2,
    logging_steps=25,
    logging_dir="./logs",
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

print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"FP16: {use_fp16}")
print()

trainer.train()

print()
print("=" * 60)
print("Training complete!")
print("=" * 60)

# ============================================================
# STEP 7: Test the model
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Testing model...")
print("=" * 60 + "\n")

model.eval()

test_messages = [
    "I feel anxious about my exams",
    "I'm so stressed with everything going on",
    "I feel hopeless and alone",
    "My parents don't understand me",
    "I've been feeling better lately",
]

for msg in test_messages:
    input_text = f"<|user|> {msg}\n<|assistant|>"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the assistant response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()

    print(f"User: {msg}")
    print(f"Bot:  {response[:200]}")
    print("-" * 60)

# ============================================================
# STEP 8: Save and download model
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Saving and downloading model...")
print("=" * 60 + "\n")

# Save model and tokenizer
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training info
training_info = {
    "base_model": MODEL_NAME,
    "training_samples": len(training_texts),
    "epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
}
with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print(f"Model saved to: {OUTPUT_DIR}")

# Zip and download
import shutil
shutil.make_archive("safemind-trained-model", "zip", ".", OUTPUT_DIR)
print("Created safemind-trained-model.zip")

from google.colab import files
files.download("safemind-trained-model.zip")

print()
print("=" * 60)
print("DONE! The model zip file is downloading.")
print("=" * 60)
```

---

## Step 4: Use the Trained Model Locally

1. Unzip `safemind-trained-model.zip` into your project:

**macOS:**
```bash
cd ~/Documents/MIDPOINT/backend
unzip ~/Downloads/safemind-trained-model.zip
```

**Windows (Git Bash):**
```bash
cd ~/Documents/MIDPOINT/backend
unzip ~/Downloads/safemind-trained-model.zip
```

2. Update your `.env` file:

```
HUGGINGFACE_MODEL=./safemind-trained-model
AI_BACKEND=local
```

3. Start the application using the [macOS Guide](GUIDE_MAC.md) or [Windows Guide](GUIDE_WINDOWS.md) Part 4.

---

## Tips for Better Accuracy

| Change | Effect |
|--------|--------|
| Increase `NUM_EPOCHS` to 7 | Model learns more (watch for overfitting) |
| Decrease `LEARNING_RATE` to 1e-5 | More stable learning |
| Generate 6000+ samples | More data = more diverse responses |
| Use `DialoGPT-large` as MODEL_NAME | Larger model = better quality (slower) |
| Increase `MAX_LENGTH` to 768 | Model handles longer conversations |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Colab disconnects mid-training | Re-run the cell from the start (re-upload dataset) |
| Out of memory | Change `BATCH_SIZE` to 4 at the top of the script |
| Loss not decreasing | Change `LEARNING_RATE` to 1e-5 |
| Gibberish responses | Train more epochs or generate more data |
| Slow training | Make sure GPU is enabled: Runtime > Change runtime type > T4 GPU |
| `No JSON file uploaded` error | Make sure you select the `.json` file when the upload dialog appears |

---

## Next Steps

- **Measure accuracy**: See [Evaluation Guide](GUIDE_EVALUATION.md)
- **Run the app**: See [macOS Guide](GUIDE_MAC.md) or [Windows Guide](GUIDE_WINDOWS.md) Part 4

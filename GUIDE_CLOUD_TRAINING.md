# SafeMind AI - Cloud Training Guide (Google Colab)

**Train your model for FREE on Google Colab with GPU acceleration.**

This guide walks you through training the SafeMind AI model entirely in the cloud using Google Colab's free GPU. This is the recommended approach if your local machine is slow or lacks a GPU.

---

## Why Cloud Training?

- **Free GPU** - Google Colab provides a free T4 GPU
- **Faster** - Training takes ~15-20 minutes vs 60-90 minutes on CPU
- **No local setup** - No need for PyTorch/CUDA on your machine
- **4000 samples** - Train with a larger dataset for better accuracy

---

## Step 1: Generate and Upload Dataset

First, generate the dataset locally:

**macOS:**
```bash
cd ~/Documents/MIDPOINT/scripts
source ../backend/venv/bin/activate
python3 expand_dataset.py --num-samples 4000 --output ../data/mental_health_dataset.json
```

**Windows:**
```cmd
cd %USERPROFILE%\Documents\MIDPOINT\scripts
..\backend\venv\Scripts\activate
python expand_dataset.py --num-samples 4000 --output ..\data\mental_health_dataset.json
```

This creates `data/mental_health_dataset.json` with 4000 training samples.

---

## Step 2: Open Google Colab

1. Go to https://colab.research.google.com
2. Sign in with your Google account
3. Click **New Notebook**
4. Go to **Runtime > Change runtime type > T4 GPU** then **Save**

---

## Step 3: Upload Your Dataset

Run this cell in Colab:

```python
from google.colab import files
uploaded = files.upload()
# Select your mental_health_dataset.json file when prompted
```

---

## Step 4: Install Dependencies

```python
!pip install transformers datasets accelerate peft torch --quiet
```

---

## Step 5: Prepare Data and Tokenize

```python
import json
from datasets import Dataset
from transformers import AutoTokenizer

# Load dataset
with open('mental_health_dataset.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data['samples'])} samples")

# Format as training text
training_texts = []
for sample in data['samples']:
    text = f"<|user|> {sample['input']}\n<|assistant|> {sample['response']}<|endoftext|>"
    training_texts.append({"text": text})

# Load tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Create HuggingFace dataset
dataset = Dataset.from_list(training_texts)

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"Tokenized {len(tokenized_dataset)} samples")
```

---

## Step 6: Train the Model

```python
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.config.pad_token_id = model.config.eos_token_id

# Training configuration - tuned for better accuracy
training_args = TrainingArguments(
    output_dir="./safemind-model",
    num_train_epochs=5,                    # More epochs for 4000 samples
    per_device_train_batch_size=8,         # Larger batch with GPU
    learning_rate=2e-5,                    # Lower LR for stability
    weight_decay=0.01,
    warmup_steps=100,
    save_steps=500,
    save_total_limit=2,
    logging_steps=25,
    fp16=True,                             # Use GPU half-precision
    report_to="none",
)

# Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
print("Starting training...")
trainer.train()
print("Training complete!")
```

**Expected output:**

```
Starting training...
{'loss': 3.21, 'learning_rate': 2e-05, 'epoch': 0.5}
{'loss': 2.14, 'learning_rate': 1.8e-05, 'epoch': 1.0}
{'loss': 1.56, 'learning_rate': 1.5e-05, 'epoch': 2.0}
{'loss': 1.12, 'learning_rate': 1.0e-05, 'epoch': 3.0}
{'loss': 0.85, 'learning_rate': 5e-06, 'epoch': 4.0}
{'loss': 0.68, 'learning_rate': 1e-06, 'epoch': 5.0}
Training complete!
```

**Watch the loss**: It should decrease each epoch. Final loss below 1.0 is good.

---

## Step 7: Test the Model in Colab

```python
import torch

model.eval()

test_messages = [
    "I feel anxious about my exams",
    "I'm so stressed with everything",
    "I feel hopeless and alone",
    "My parents don't understand me",
    "I've been feeling better lately",
]

print("=" * 60)
print("MODEL TEST RESULTS")
print("=" * 60)

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
    response = response.split("<|assistant|>")[-1].strip()

    print(f"\nUser: {msg}")
    print(f"Bot:  {response[:200]}")
    print("-" * 60)
```

---

## Step 8: Save and Download Model

```python
# Save the model
OUTPUT_DIR = "./safemind-trained-model"
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Zip the model folder
!zip -r safemind-trained-model.zip safemind-trained-model/

# Download it
from google.colab import files
files.download('safemind-trained-model.zip')
```

---

## Step 9: Use the Model Locally

1. Unzip `safemind-trained-model.zip` into your project:

**macOS:**
```bash
cd ~/Documents/MIDPOINT/backend
unzip ~/Downloads/safemind-trained-model.zip
```

**Windows:**
Extract the zip file to `MIDPOINT\backend\safemind-trained-model\`

2. Update `.env`:

```
HUGGINGFACE_MODEL=./safemind-trained-model
AI_BACKEND=local
```

3. Start the application using the [macOS Guide](GUIDE_MAC.md) or [Windows Guide](GUIDE_WINDOWS.md) Part 4.

---

## Tips for Better Accuracy

| Change | Effect |
|--------|--------|
| Increase `num_train_epochs` to 7 | Model learns more (watch for overfitting) |
| Decrease `learning_rate` to 1e-5 | More stable learning |
| Generate 6000+ samples | More data = more diverse responses |
| Use `DialoGPT-large` as MODEL_NAME | Larger model = better quality (slower) |
| Increase `max_length` to 768 | Model handles longer conversations |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Colab disconnects | Re-run cells from Step 4 onward, dataset stays uploaded |
| Out of memory | Reduce `per_device_train_batch_size` to 4 |
| Loss not decreasing | Lower learning rate to 1e-5 |
| Gibberish responses | Train more epochs or use more data |
| Slow training | Make sure GPU is enabled: Runtime > Change runtime type > T4 GPU |

---

## Next Steps

- **Measure accuracy**: See [Evaluation Guide](GUIDE_EVALUATION.md)
- **Run the app**: See [macOS Guide](GUIDE_MAC.md) or [Windows Guide](GUIDE_WINDOWS.md) Part 4

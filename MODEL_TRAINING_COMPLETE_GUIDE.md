# SafeMind AI - Complete Model Training Guide

**Version:** 2.0
**Date:** January 5, 2026
**Difficulty:** Intermediate
**Estimated Time:** 2-4 hours (dataset generation) + 1-2 hours (training)

---

## Table of Contents

1. [Training Overview](#1-training-overview)
2. [Prerequisites](#2-prerequisites)
3. [Synthetic Dataset Generation](#3-synthetic-dataset-generation)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Model Selection](#5-model-selection)
6. [LoRA Fine-Tuning Setup](#6-lora-fine-tuning-setup)
7. [Training Process](#7-training-process)
8. [Model Evaluation](#8-model-evaluation)
9. [Model Integration](#9-model-integration)
10. [Troubleshooting](#10-troubleshooting)
11. [Advanced Topics](#11-advanced-topics)

---

## 1. Training Overview

### 1.1 What You'll Learn

This guide teaches you to train a custom mental health chatbot model from scratch:

1. **Generate synthetic training data** using an external LLM (GPT/Claude/Gemini)
2. **Prepare dataset** in proper format for training
3. **Fine-tune a base LLM** using LoRA (Low-Rank Adaptation)
4. **Evaluate model performance** with test cases
5. **Integrate trained model** into the SafeMind AI system

### 1.2 Training Approach

We use **LoRA (Low-Rank Adaptation)** fine-tuning because:

- ✅ Works on consumer hardware (no expensive GPUs needed)
- ✅ Fast training (30-60 minutes vs. days for full fine-tuning)
- ✅ Small file size (~10-50MB adapters vs. 500MB+ full models)
- ✅ Easy to swap and test different adaptations
- ✅ FREE using Google Colab

### 1.3 Training Pipeline

```
Step 1: Generate Synthetic Data
   ↓ (Use GPT/Claude/Gemini API)
   └→ 1000-3000 conversations in JSON

Step 2: Validate & Clean Data
   ↓ (Manual review + automated checks)
   └→ High-quality training dataset

Step 3: Prepare Training Format
   ↓ (Convert to instruction format)
   └→ Tokenized dataset

Step 4: LoRA Fine-Tuning
   ↓ (Train on GPU - Colab or local)
   └→ LoRA adapter weights

Step 5: Merge & Save Model
   ↓ (Combine base + adapter)
   └→ Production-ready model

Step 6: Evaluate & Test
   ↓ (Run test cases)
   └→ Verified model performance

Step 7: Deploy to System
   └→ Integrated with SafeMind AI
```

---

## 2. Prerequisites

### 2.1 Required Knowledge

- Basic Python programming
- Understanding of command line
- Familiarity with JSON format
- Basic understanding of LLMs (what they are, how they work)

### 2.2 Required Software

```bash
# Python 3.9+
python --version

# Required libraries
pip install transformers datasets torch peft accelerate trl
pip install sentencepiece protobuf

# For dataset generation
pip install openai anthropic google-generativeai  # Choose one
```

### 2.3 Required Accounts

**For Dataset Generation (choose one):**

- **OpenAI Account** ($5 credit recommended)
  - Sign up: https://platform.openai.com/
  - API: ~$0.50 for 1000 samples

- **Anthropic Account** (Claude - recommended for quality)
  - Sign up: https://console.anthropic.com/
  - API: ~$1.50 for 1000 samples

- **Google AI Studio** (Gemini - free tier available)
  - Sign up: https://ai.google.dev/

**For Training:**

- **Google Colab** (free GPU)
  - Sign up: https://colab.research.google.com/
  - OR local GPU (NVIDIA with CUDA)

### 2.4 Hardware Requirements

**For Dataset Generation:**
- Any computer with internet

**For Training:**

**Option A: Google Colab (FREE)**
- No local requirements
- Free T4 GPU (15GB VRAM)
- ~50GB free storage

**Option B: Local Training**
- NVIDIA GPU with 8GB+ VRAM (recommended)
- OR CPU-only (slower but works)
- 16GB RAM
- 10GB free disk space

---

## 3. Synthetic Dataset Generation

### 3.1 Why Synthetic Data?

There is **no existing dataset** of Sri Lankan mental health conversations. We must create one.

**Benefits of synthetic data:**
- ✅ Control over quality and diversity
- ✅ Cultural context customization
- ✅ Privacy-preserving (no real user data)
- ✅ Scalable (generate thousands of samples)

### 3.2 Dataset Generation Script

Create `scripts/generate_dataset.py`:

```python
"""
SafeMind AI - Synthetic Dataset Generator
Generates culturally-aware mental health training conversations
"""

import json
import os
import time
from typing import List, Dict
import random

# Choose your API (uncomment one)
# from openai import OpenAI  # OpenAI GPT
from anthropic import Anthropic  # Claude (recommended)
# import google.generativeai as genai  # Google Gemini

class DatasetGenerator:
    def __init__(self, api_key: str, provider: str = "claude"):
        """
        Initialize dataset generator

        Args:
            api_key: API key for LLM provider
            provider: "openai", "claude", or "gemini"
        """
        self.provider = provider

        if provider == "claude":
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-5-sonnet-20241022"
        elif provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4"
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')

    def generate_sample(self, user_input: str, category: str,
                       emotion: str, risk_level: str) -> Dict:
        """
        Generate a single training sample

        Args:
            user_input: Example user message (seed)
            category: anxiety, depression, stress, crisis, etc.
            emotion: sad, anxious, hopeless, stressed, neutral
            risk_level: low, medium, high

        Returns:
            Dict with instruction, input, response, emotion, risk_level
        """

        prompt = f"""Generate a mental health chatbot training sample for the Sri Lankan context.

CONTEXT:
- User message category: {category}
- Emotion: {emotion}
- Risk level: {risk_level}
- Cultural context: Sri Lanka (family pressure, academic stress, social stigma)

USER MESSAGE (example):
"{user_input}"

Generate a SIMILAR but DIFFERENT user message and appropriate response.

REQUIREMENTS FOR RESPONSE:
1. DO NOT diagnose any mental illness
2. DO NOT provide medical or therapeutic advice
3. Use empathetic, supportive tone
4. Reflect Sri Lankan cultural context:
   - Family expectations and pressure
   - Academic stress (A/L exams, university)
   - Social stigma around mental health
   - Financial concerns
5. Encourage self-reflection
6. Suggest professional help gently when appropriate
7. Use Sri Lankan English (respectful, neutral)
8. If high risk: Include crisis resources (1333 hotline)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "instruction": "You are a mental health awareness chatbot for Sri Lankan users. Provide empathetic, culturally-aware support without diagnosing or replacing professional care.",
  "input": "[Generated user message - make it realistic and different from example]",
  "response": "[Your empathetic, culturally-appropriate response]",
  "emotion": "{emotion}",
  "risk_level": "{risk_level}"
}}
"""

        try:
            if self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content

            elif self.provider == "gemini":
                response = self.model.generate_content(prompt)
                content = response.text

            # Parse JSON from response
            # Remove markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            sample = json.loads(content)

            # Validate required fields
            required_fields = ["instruction", "input", "response", "emotion", "risk_level"]
            if all(field in sample for field in required_fields):
                return sample
            else:
                print(f"Warning: Missing fields in generated sample")
                return None

        except Exception as e:
            print(f"Error generating sample: {e}")
            return None

    def generate_dataset(self, num_samples: int = 1000,
                        output_file: str = "../data/synthetic_training_data.json") -> List[Dict]:
        """
        Generate complete dataset

        Args:
            num_samples: Number of samples to generate
            output_file: Where to save the dataset

        Returns:
            List of generated samples
        """

        # Seed examples for each category
        seeds = {
            "anxiety": [
                ("I'm very nervous about my A/L results", "anxious", "medium"),
                ("I can't sleep because I'm worried about the interview", "anxious", "medium"),
                ("My heart races when I think about exams", "anxious", "low"),
            ],
            "depression": [
                ("I feel sad all the time and nothing makes me happy", "sad", "medium"),
                ("I have no energy to do anything anymore", "sad", "medium"),
                ("Everything feels meaningless lately", "sad", "high"),
            ],
            "family_pressure": [
                ("My parents want me to be a doctor but I want to study arts", "stressed", "low"),
                ("Everyone compares me to my cousin who went abroad", "sad", "medium"),
                ("I feel like I'm disappointing my family", "sad", "medium"),
            ],
            "academic_stress": [
                ("I failed my university exam and don't know what to do", "stressed", "medium"),
                ("There's too much pressure to get high marks", "anxious", "low"),
                ("I can't handle the workload at university", "stressed", "medium"),
            ],
            "financial_stress": [
                ("My family is struggling financially and I feel guilty", "stressed", "medium"),
                ("I can't find a job after graduation", "sad", "medium"),
                ("I worry about money all the time", "anxious", "low"),
            ],
            "crisis": [
                ("I don't see the point in living anymore", "hopeless", "high"),
                ("I feel completely hopeless about everything", "hopeless", "high"),
                ("I want everything to end", "hopeless", "high"),
            ],
            "relationship": [
                ("I feel very lonely and have no one to talk to", "sad", "medium"),
                ("My friends don't understand what I'm going through", "sad", "low"),
                ("I feel isolated from everyone", "sad", "medium"),
            ],
            "positive": [
                ("I tried the breathing exercises and they helped", "neutral", "low"),
                ("I'm feeling a bit better after talking", "neutral", "low"),
                ("Thank you for listening to me", "neutral", "low"),
            ]
        }

        dataset = []
        categories = list(seeds.keys())

        print("=" * 60)
        print(f"Generating {num_samples} synthetic training samples...")
        print("=" * 60)

        for i in range(num_samples):
            # Select category (weighted to avoid too many crisis samples)
            if i % 10 == 0:  # 10% crisis samples
                category = "crisis"
            else:
                category = random.choice([c for c in categories if c != "crisis"])

            # Get random seed from category
            seed_data = random.choice(seeds[category])
            user_input, emotion, risk_level = seed_data

            # Generate sample
            print(f"\n[{i+1}/{num_samples}] Generating {category} sample...")
            sample = self.generate_sample(user_input, category, emotion, risk_level)

            if sample:
                sample["category"] = category
                dataset.append(sample)
                print(f"✓ Generated: {sample['input'][:60]}...")
            else:
                print(f"✗ Failed to generate sample")

            # Rate limiting (be nice to APIs)
            time.sleep(1)  # 1 second between requests

            # Save checkpoint every 100 samples
            if (i + 1) % 100 == 0:
                self._save_checkpoint(dataset, output_file)
                print(f"\n✓ Checkpoint: {len(dataset)} samples saved")

        # Final save
        self._save_dataset(dataset, output_file)

        print("\n" + "=" * 60)
        print(f"✓ Dataset generation complete!")
        print(f"✓ Total samples: {len(dataset)}")
        print(f"✓ Saved to: {output_file}")
        print("=" * 60)

        return dataset

    def _save_checkpoint(self, dataset: List[Dict], output_file: str):
        """Save intermediate checkpoint"""
        checkpoint_file = output_file.replace(".json", "_checkpoint.json")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "samples": dataset,
                "count": len(dataset),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2, ensure_ascii=False)

    def _save_dataset(self, dataset: List[Dict], output_file: str):
        """Save final dataset"""
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_samples": len(dataset),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "provider": self.provider,
                    "model": self.model
                },
                "samples": dataset
            }, f, indent=2, ensure_ascii=False)


def main():
    """Main execution"""

    # Configuration
    API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Change based on provider
    PROVIDER = "claude"  # "openai", "claude", or "gemini"
    NUM_SAMPLES = 1000  # Start with 1000, can generate more later
    OUTPUT_FILE = "../data/synthetic_training_data.json"

    if not API_KEY:
        print("Error: API key not found!")
        print("Set environment variable: export ANTHROPIC_API_KEY=your-key")
        return

    # Initialize generator
    generator = DatasetGenerator(api_key=API_KEY, provider=PROVIDER)

    # Generate dataset
    dataset = generator.generate_dataset(
        num_samples=NUM_SAMPLES,
        output_file=OUTPUT_FILE
    )

    # Print statistics
    categories = {}
    emotions = {}
    risk_levels = {}

    for sample in dataset:
        cat = sample.get("category", "unknown")
        emo = sample.get("emotion", "unknown")
        risk = sample.get("risk_level", "unknown")

        categories[cat] = categories.get(cat, 0) + 1
        emotions[emo] = emotions.get(emo, 0) + 1
        risk_levels[risk] = risk_levels.get(risk, 0) + 1

    print("\n📊 Dataset Statistics:")
    print(f"\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({count/len(dataset)*100:.1f}%)")

    print(f"\nEmotions:")
    for emo, count in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"  {emo}: {count} ({count/len(dataset)*100:.1f}%)")

    print(f"\nRisk Levels:")
    for risk, count in sorted(risk_levels.items(), key=lambda x: -x[1]):
        print(f"  {risk}: {count} ({count/len(dataset)*100:.1f}%)")


if __name__ == "__main__":
    main()
```

### 3.3 Run Dataset Generation

```bash
# Set API key
export ANTHROPIC_API_KEY=your-key-here  # Or OPENAI_API_KEY

# Run generator
cd scripts
python generate_dataset.py
```

**Expected output:**
```
============================================================
Generating 1000 synthetic training samples...
============================================================

[1/1000] Generating anxiety sample...
✓ Generated: I'm feeling very anxious about my upcoming university...

[2/1000] Generating family_pressure sample...
✓ Generated: My parents keep comparing me to my brother who...

...

✓ Checkpoint: 100 samples saved
...

============================================================
✓ Dataset generation complete!
✓ Total samples: 982
✓ Saved to: ../data/synthetic_training_data.json
============================================================

📊 Dataset Statistics:

Categories:
  anxiety: 205 (20.9%)
  family_pressure: 198 (20.2%)
  academic_stress: 187 (19.0%)
  ...
```

**Time estimate:** ~1-2 hours for 1000 samples (depends on API rate limits)

**Cost estimate:**
- OpenAI GPT-4: ~$0.50-$1.00
- Claude Sonnet: ~$1.50-$2.00
- Gemini: Free (with rate limits)

---

## 4. Dataset Preparation

### 4.1 Manual Validation

Review ~15% of samples manually:

```python
# Load dataset
with open('../data/synthetic_training_data.json', 'r') as f:
    data = json.load(f)

samples = data['samples']

# Review random samples
import random
review_samples = random.sample(samples, min(150, len(samples)))

for i, sample in enumerate(review_samples):
    print(f"\n=== Sample {i+1} ===")
    print(f"Input: {sample['input']}")
    print(f"Response: {sample['response']}")
    print(f"Category: {sample['category']} | Risk: {sample['risk_level']}")

    # Check for issues
    response_lower = sample['response'].lower()

    # Flag potential issues
    issues = []
    if any(word in response_lower for word in ['diagnose', 'medication', 'prescription']):
        issues.append("⚠️ Contains medical advice")
    if 'you should' in response_lower:
        issues.append("⚠️ Too directive")
    if len(sample['response']) < 50:
        issues.append("⚠️ Response too short")

    if issues:
        print(f"ISSUES: {', '.join(issues)}")
        # Mark for removal or editing
```

### 4.2 Clean Dataset

Remove problematic samples:

```python
def clean_dataset(samples):
    """Remove low-quality samples"""
    cleaned = []
    removed = 0

    for sample in samples:
        response = sample['response'].lower()

        # Removal criteria
        if any([
            len(sample['response']) < 30,  # Too short
            'i am an ai' in response,  # Meta references
            'i cannot' in response and sample['risk_level'] == 'low',  # Over-cautious
            any(word in response for word in ['diagnose', 'prescribe', 'medication']),
        ]):
            removed += 1
            continue

        cleaned.append(sample)

    print(f"✓ Cleaned: {len(cleaned)} samples (removed {removed})")
    return cleaned

# Clean and save
cleaned_samples = clean_dataset(samples)

with open('../data/training_data_cleaned.json', 'w') as f:
    json.dump({
        "metadata": {
            "total_samples": len(cleaned_samples),
            "cleaned_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "samples": cleaned_samples
    }, f, indent=2)
```

### 4.3 Convert to Training Format

```python
def format_for_training(sample):
    """Convert to instruction format"""
    return {
        "text": f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['response']}"""
    }

# Convert all samples
training_formatted = [format_for_training(s) for s in cleaned_samples]

# Save
with open('../data/training_data_formatted.json', 'w') as f:
    json.dump(training_formatted, f, indent=2)
```

---

## 5. Model Selection

### 5.1 Recommended Base Models

| Model | Size | Pros | Cons | Recommended? |
|-------|------|------|------|--------------|
| **microsoft/phi-3-mini-4k** | 3.8B | Excellent quality, efficient | Newer, less tested | ⭐⭐⭐⭐⭐ |
| **microsoft/DialoGPT-medium** | 345M | Fast, conversational | Older, less capable | ⭐⭐⭐⭐ |
| **meta-llama/Llama-3.2-1B** | 1B | High quality, efficient | May need approval | ⭐⭐⭐⭐ |
| **google/flan-t5-base** | 250M | Fast, general | Not chat-optimized | ⭐⭐⭐ |

**Recommendation:** Use **Phi-3 Mini** for best balance of quality and efficiency.

### 5.2 Test Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-3-mini-4k-instruct"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Test generation
test_input = "I feel anxious about my exams"
inputs = tokenizer(test_input, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {test_input}")
print(f"Output: {response}")
```

---

## 6. LoRA Fine-Tuning Setup

### 6.1 Install LoRA Dependencies

```bash
pip install peft transformers accelerate datasets trl bitsandbytes
```

### 6.2 Enhanced Training Script with LoRA

Create `backend/train_model_lora.py`:

```python
"""
SafeMind AI - LoRA Fine-Tuning Script
Efficient fine-tuning using LoRA (Low-Rank Adaptation)
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
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

print("=" * 60)
print("SafeMind AI - LoRA Fine-Tuning")
print("=" * 60)

# Configuration
BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"  # Change as needed
DATASET_PATH = "../data/training_data_formatted.json"
OUTPUT_DIR = "./safemind-lora-model"
LORA_R = 8  # LoRA rank
LORA_ALPHA = 16  # LoRA scaling
LORA_DROPOUT = 0.1

# Step 1: Load Dataset
print("\n[1/7] Loading training dataset...")
with open(DATASET_PATH, 'r') as f:
    training_data = json.load(f)

print(f"✓ Loaded {len(training_data)} training samples")

# Step 2: Load Base Model and Tokenizer
print(f"\n[2/7] Loading base model: {BASE_MODEL}")
print("This may take a few minutes on first run...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

print("✓ Base model loaded")

# Step 3: Configure LoRA
print("\n[3/7] Configuring LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,  # Rank
    lora_alpha=LORA_ALPHA,  # Scaling factor
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✓ LoRA configuration applied")

# Step 4: Prepare Dataset
print("\n[4/7] Preparing dataset...")

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=512,
        padding='max_length'
    )

# Create HuggingFace dataset
dataset = Dataset.from_list(training_data)
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=['text'],
    batched=True
)

print(f"✓ Prepared {len(tokenized_dataset)} tokenized examples")

# Step 5: Training Configuration
print("\n[5/7] Setting up training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,  # Mixed precision
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    logging_dir='./logs',
    report_to="none",
    warmup_steps=50,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("✓ Training configuration ready")

# Step 6: Train
print("\n[6/7] Starting training...")
print("=" * 60)
print("Expected time: 30-60 minutes (depends on GPU)")
print("Monitor loss values - lower is better!")
print("=" * 60)
print()

trainer.train()

print("\n" + "=" * 60)
print("✓ Training complete!")
print("=" * 60)

# Step 7: Save Model
print("\n[7/7] Saving fine-tuned model...")

# Save LoRA adapters
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ SUCCESS! Model saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("1. Test model with test_trained_model.py")
print("2. Update .env: AI_BACKEND=local")
print("3. Update .env: LOCAL_MODEL=./safemind-lora-model")
print("4. Run: python app_improved.py")
```

---

## 7. Training Process

### 7.1 Local Training (with GPU)

```bash
cd backend
python train_model_lora.py
```

### 7.2 Google Colab Training (FREE GPU)

**Step 1:** Go to https://colab.research.google.com/

**Step 2:** Create new notebook

**Step 3:** Add cells:

```python
# Cell 1: Install dependencies
!pip install transformers datasets peft accelerate trl bitsandbytes

# Cell 2: Upload files
from google.colab import files
uploaded = files.upload()  # Upload training_data_formatted.json

# Cell 3: Run training script
# Copy entire train_model_lora.py content here
# ...

# Cell 4: Download trained model
!zip -r safemind-lora-model.zip safemind-lora-model
files.download('safemind-lora-model.zip')
```

**Step 4:** Runtime → Change runtime type → T4 GPU

**Step 5:** Run all cells

### 7.3 Monitor Training

**Expected output:**
```
{'loss': 2.456, 'learning_rate': 0.0002, 'epoch': 0.5}
{'loss': 1.823, 'learning_rate': 0.00015, 'epoch': 1.0}
{'loss': 1.456, 'learning_rate': 0.0001, 'epoch': 1.5}
{'loss': 1.234, 'learning_rate': 0.00005, 'epoch': 2.0}
{'loss': 1.123, 'learning_rate': 0.00001, 'epoch': 2.5}
{'loss': 1.089, 'learning_rate': 0.0, 'epoch': 3.0}
```

**Good signs:**
- ✅ Loss decreases steadily (2.5 → 1.0)
- ✅ No NaN or Inf values
- ✅ Completes without errors

**Bad signs:**
- ❌ Loss increases
- ❌ Loss stays constant
- ❌ NaN/Inf values (reduce learning rate)
- ❌ Out of memory (reduce batch size)

---

## 8. Model Evaluation

### 8.1 Test Script

Create `backend/test_trained_model.py`:

```python
"""Test the trained model"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./safemind-lora-model")
tokenizer = AutoTokenizer.from_pretrained("./safemind-lora-model")

# Test cases
test_cases = [
    "I feel very anxious about my A/L results",
    "My parents want me to be an engineer but I want to study art",
    "I feel like I'm not good enough",
    "I feel hopeless about everything",
]

print("\n" + "=" * 60)
print("Testing Trained Model")
print("=" * 60)

for i, test_input in enumerate(test_cases, 1):
    print(f"\n[Test {i}]")
    print(f"Input: {test_input}")

    # Format with instruction
    prompt = f"""### Instruction:
You are a mental health awareness chatbot. Provide empathetic support.

### Input:
{test_input}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    response = response.split("### Response:")[-1].strip()

    print(f"Output: {response}")
    print("-" * 60)
```

**Run tests:**
```bash
python test_trained_model.py
```

### 8.2 Evaluation Metrics

**Qualitative:**
- Empathy and tone
- Cultural relevance
- Avoidance of diagnosis/medical advice
- Appropriate crisis handling

**Quantitative (optional):**
```python
from datasets import load_metric

bleu = load_metric("bleu")
rouge = load_metric("rouge")

# Compare against reference responses
# Calculate BLEU/ROUGE scores
```

---

## 9. Model Integration

### 9.1 Update Backend Configuration

Edit `backend/.env`:
```env
AI_BACKEND=local
LOCAL_MODEL=./safemind-lora-model
```

### 9.2 Update Model Loading

Edit `backend/ai_model_free.py` to support LoRA:

```python
from peft import PeftModel

def load_local_model(self):
    if os.path.exists(os.path.join(self.model_name, "adapter_config.json")):
        # LoRA model
        base_name = json.load(open(os.path.join(self.model_name, "adapter_config.json")))['base_model_name_or_path']
        base_model = AutoModelForCausalLM.from_pretrained(base_name)
        self.model = PeftModel.from_pretrained(base_model, self.model_name)
    else:
        # Regular model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
```

### 9.3 Test Integration

```bash
cd backend
python app_improved.py
```

Test via API:
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I feel anxious about my exams",
    "session_id": "test123"
  }'
```

---

## 10. Troubleshooting

### Issue: CUDA Out of Memory

**Solutions:**
```python
# Reduce batch size
per_device_train_batch_size=2  # or 1

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use 8-bit training
load_in_8bit=True
```

### Issue: Loss Not Decreasing

**Solutions:**
- Increase learning rate: `2e-4` → `5e-4`
- Increase LoRA rank: `r=8` → `r=16`
- Check dataset quality
- Train for more epochs

### Issue: Model Generates Gibberish

**Solutions:**
- Check tokenizer padding
- Verify dataset format
- Reduce temperature during generation
- Use beam search instead of sampling

### Issue: Import Errors

```bash
pip install --upgrade transformers peft accelerate
```

---

## 11. Advanced Topics

### 11.1 Hyperparameter Tuning

```python
# Experiment with these
learning_rates = [1e-4, 2e-4, 5e-4]
lora_ranks = [4, 8, 16, 32]
epochs = [2, 3, 5]
```

### 11.2 Multi-GPU Training

```python
training_args = TrainingArguments(
    # ...
    ddp_find_unused_parameters=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
)
```

### 11.3 Merging LoRA Adapters

```python
# Merge adapters into base model (optional)
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
peft_model = PeftModel.from_pretrained(base_model, "./safemind-lora-model")

# Merge
merged_model = peft_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./safemind-merged-model")
```

### 11.4 Quantization for Deployment

```python
# 4-bit quantization for smaller size
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

---

## Summary Checklist

**Dataset Generation:**
- [ ] Install API libraries
- [ ] Get API key (OpenAI/Claude/Gemini)
- [ ] Run dataset generator
- [ ] Generate 1000+ samples
- [ ] Manually validate samples
- [ ] Clean dataset
- [ ] Format for training

**Model Training:**
- [ ] Install training dependencies
- [ ] Choose base model
- [ ] Configure LoRA settings
- [ ] Set up training environment (Colab or local)
- [ ] Run training script
- [ ] Monitor training progress
- [ ] Save trained model

**Evaluation:**
- [ ] Test with sample inputs
- [ ] Verify response quality
- [ ] Check cultural relevance
- [ ] Test crisis handling
- [ ] Calculate metrics (optional)

**Integration:**
- [ ] Update backend config
- [ ] Load LoRA model
- [ ] Test API endpoints
- [ ] Verify full system works

---

**Congratulations!** 🎉

You now have a custom-trained mental health chatbot model fine-tuned for Sri Lankan context!

---

**Document Version:** 2.0
**Last Updated:** January 5, 2026

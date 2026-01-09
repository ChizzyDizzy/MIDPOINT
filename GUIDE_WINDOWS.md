# SafeMind AI - Complete Setup Guide (Windows)

**Sri Lankan Mental Health Chatbot - Final Year Project**
**Student:** Chirath Sanduwara Wijesinghe (CB011568)

---

## 🎯 What You'll Build

A complete mental health chatbot powered by:
- **Template expansion** (NO API needed!) for dataset generation
- **Hugging Face models** (DialoGPT/GPT-2) for the chatbot
- **LoRA fine-tuning** for Sri Lankan context
- **React frontend** for the user interface
- **9-layer crisis detection** system

**NO Gemini. NO Claude. NO OpenAI. NO API keys. Pure Hugging Face!**

---

## 📋 Prerequisites

### Required Software

**1. Python 3.9+**
1. Download from https://www.python.org/downloads/
2. **IMPORTANT:** Check "Add Python to PATH" during installation
3. Verify:
```cmd
python --version
```

**2. Node.js 16+**
1. Download from https://nodejs.org/
2. Install LTS version
3. Verify:
```cmd
node --version
npm --version
```

**3. Git**
1. Download from https://git-scm.com/download/win
2. Install with default settings
3. Verify:
```cmd
git --version
```

### Required Accounts

**Hugging Face Account (FREE - for models)**
1. Go to https://huggingface.co/join
2. Sign up (FREE)
3. Go to Settings → Access Tokens
4. Create token (read access)
5. Copy token (starts with `hf_`)

---

## 🚀 PART 1: Setup Backend & Frontend (30 minutes)

### Step 1: Clone Repository

```cmd
cd C:\Users\%USERNAME%\Documents
git clone https://github.com/ChizzyDizzy/MIDPOINT.git
cd MIDPOINT
```

### Step 2: Backend Setup

```cmd
cd backend

:: Create virtual environment
python -m venv venv

:: Activate it
venv\Scripts\activate

:: Install dependencies
python -m pip install --upgrade pip
pip install torch transformers datasets accelerate peft
pip install flask flask-cors python-dotenv textblob requests numpy pandas scikit-learn nltk

:: Download NLP data
python -m textblob.download_corpora
```

### Step 3: Frontend Setup

```cmd
cd ..\frontend

:: Install dependencies
npm install
```

### Step 4: Configure Backend

```cmd
cd ..\backend

:: Create environment file
copy .env.example .env

:: Edit .env
notepad .env
```

**Add to `.env`:**
```env
# Hugging Face Configuration
HUGGINGFACE_API_KEY=hf_your_token_here
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium

# AI Backend
AI_BACKEND=local

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development
FLASK_DEBUG=True

# Sri Lankan Resources
CRISIS_DETECTION_THRESHOLD=0.7
ENABLE_AI_RESPONSES=True
DEFAULT_CULTURE=south_asian
```

**Generate secret key:**
```cmd
python -c "import secrets; print(secrets.token_hex(32))"
:: Copy output to FLASK_SECRET_KEY
```

---

## 📊 PART 2: Generate Synthetic Dataset (1 minute - NO API needed!)

### Generate 1500 Training Samples

**NO API keys needed!** Uses existing template files in `data\` directory.

```cmd
cd ..\scripts

:: Generate 1500 samples from templates (< 1 minute!)
python expand_dataset.py --num-samples 1500 --output ..\data\mental_health_dataset.json
```

**Expected Output:**
```
======================================================================
Expanding templates into 1500 training samples...
======================================================================

[100/1500] Generated anxiety sample
[200/1500] Generated anxiety sample
...
[1500/1500] Generated positive sample

✓ Generated 1500 training samples

======================================================================
✓ Dataset expansion complete!
✓ Total samples: 1500
✓ Saved to: ..\data\mental_health_dataset.json
======================================================================

📊 Dataset Statistics:

Total Samples: 1500

Categories:
  anxiety               250 ( 16.7%)
  academic_stress       200 ( 13.3%)
  stress                200 ( 13.3%)
  depression            200 ( 13.3%)
  family_issues         150 ( 10.0%)
  ...
```

**What this does:**
- Loads existing JSON templates (crisis patterns, cultural templates, etc.)
- Generates culturally-aware Sri Lankan variations
- Creates 1500 properly formatted training samples
- Includes proper risk levels and crisis resources
- NO external API calls needed!

### Step 3: Validate Dataset

```cmd
cd ..\backend
python
```

**In Python:**
```python
import json

with open('../data/mental_health_dataset.json', 'r') as f:
    data = json.load(f)

samples = data['samples']
print(f"✓ Total samples: {len(samples)}")
print(f"\nFirst sample:")
print(f"Input: {samples[0]['input']}")
print(f"Response: {samples[0]['response'][:150]}...")
print(f"Category: {samples[0]['category']}")
print(f"Risk: {samples[0]['risk_level']}")

# Show statistics
categories = {}
for s in samples:
    cat = s.get('category', 'unknown')
    categories[cat] = categories.get(cat, 0) + 1

print("\nCategories:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

exit()
```

---

## 🤖 PART 3: Train Your Model (60-90 minutes)

### Step 1: Create Training Script

```cmd
cd ..\backend
notepad train_model.py
```

**Add this content and save:**
```python
"""
Train Hugging Face model on mental health dataset
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

# Configuration
MODEL_NAME = "microsoft/DialoGPT-medium"  # or "gpt2"
DATASET_PATH = "../data/mental_health_dataset.json"
OUTPUT_DIR = "./mental_health_model"
MAX_LENGTH = 512

print("=" * 70)
print("SafeMind AI - Model Training")
print("=" * 70)

# Load dataset
print("\n[1/5] Loading dataset...")
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

# Format data for training
print("[2/5] Formatting data...")
training_texts = []
for sample in data['samples']:
    text = f"""<|user|> {sample['input']}
<|assistant|> {sample['response']}<|endoftext|>"""
    training_texts.append({"text": text})

print(f"✓ Prepared {len(training_texts)} training examples")

# Load model and tokenizer
print("\n[3/5] Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print(f"✓ Loaded {MODEL_NAME}")

# Tokenize dataset
print("\n[4/5] Tokenizing dataset...")
dataset = Dataset.from_list(training_texts)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print(f"✓ Tokenized {len(tokenized_dataset)} samples")

# Training arguments
print("\n[5/5] Starting training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# Train!
trainer.train()

# Save
print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 70)
print("✓ Training complete!")
print(f"✓ Model saved to: {OUTPUT_DIR}")
print("=" * 70)
```

### Step 2: Run Training

```cmd
python train_model.py
```

**This will take 60-90 minutes on Windows.**

---

## 🎯 PART 4: Fine-Tune with LoRA (Optional - 30 minutes)

```cmd
notepad fine_tune_lora.py
```

**Add and save:**
```python
"""
Fine-tune model using LoRA for efficiency
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_NAME = "microsoft/DialoGPT-medium"
DATASET_PATH = "../data/mental_health_dataset.json"
OUTPUT_DIR = "./mental_health_model_lora"

print("Loading model with LoRA...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

training_texts = []
for sample in data['samples']:
    text = f"<|user|> {sample['input']}\n<|assistant|> {sample['response']}<|endoftext|>"
    training_texts.append({"text": text})

dataset = Dataset.from_list(training_texts)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=3e-4,
    save_steps=500,
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

print("Training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✓ LoRA model saved to: {OUTPUT_DIR}")
```

**Run:**
```cmd
python fine_tune_lora.py
```

---

## 🔌 PART 5: Connect Model to Backend (15 minutes)

```cmd
notepad chatbot_app.py
```

**Add and save:**
```python
"""
SafeMind AI Chatbot Backend
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = "./mental_health_model"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()
print("✓ Model loaded!")

CRISIS_KEYWORDS = {
    'immediate': ['kill myself', 'suicide', 'end my life', 'want to die'],
    'high': ['hopeless', 'no point living', 'burden to everyone'],
    'medium': ['worthless', 'hate myself', 'can\'t go on']
}

def detect_crisis(text):
    text_lower = text.lower()
    for word in CRISIS_KEYWORDS['immediate']:
        if word in text_lower:
            return 'immediate'
    for word in CRISIS_KEYWORDS['high']:
        if word in text_lower:
            return 'high'
    for word in CRISIS_KEYWORDS['medium']:
        if word in text_lower:
            return 'medium'
    return 'low'

def add_crisis_response(response, level):
    if level == 'immediate':
        return response + "\n\n🚨 **IMMEDIATE HELP:**\n- National Crisis Hotline (Sri Lanka): **1333**\n- Emergency Services: **119**\n- Sumithrayo: **011-2696666**"
    elif level == 'high':
        return response + "\n\n**Support Resources:**\n- Mental Health Helpline: **1926**\n- Sumithrayo: **011-2696666**"
    return response

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message'}), 400

    crisis_level = detect_crisis(user_message)

    input_text = f"<|user|> {user_message}\n<|assistant|>"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|assistant|>")[-1].strip()
    response = add_crisis_response(response, crisis_level)

    return jsonify({
        'response': response,
        'crisis_level': crisis_level,
        'timestamp': str(datetime.datetime.now())
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting SafeMind AI...")
    print("API: http://localhost:5000")
    app.run(debug=True, port=5000)
```

**Run:**
```cmd
python chatbot_app.py
```

**Test (in another CMD):**
```cmd
curl -X POST http://localhost:5000/api/chat -H "Content-Type: application/json" -d "{\"message\": \"I feel stressed\"}"
```

---

## 🎨 PART 6: Connect Frontend (15 minutes)

```cmd
cd ..\frontend\src\services
notepad api.js
```

**Ensure it has:**
```javascript
const API_URL = 'http://localhost:5000/api';

export const sendMessage = async (message) => {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  });
  return response.json();
};
```

**Start frontend:**
```cmd
cd ..\..
npm start
```

---

## ✅ PART 7: Testing (10 minutes)

1. **Open:** http://localhost:3000
2. **Test normal:** "I'm stressed about exams"
3. **Test crisis:** "I feel hopeless"
4. **Test cultural:** "My parents want me to be a doctor"

---

## 🎉 Complete!

✅ Mental health chatbot
✅ Sri Lankan context
✅ Crisis detection
✅ Working app
✅ FREE to run

**Your final year project is done!** 🚀

---

## 🆘 Troubleshooting

**"Python not found"**
- Reinstall Python with "Add to PATH" checked

**"Module not found"**
- Activate venv: `venv\Scripts\activate`
- Install: `pip install [module]`

**"Port already in use"**
- Kill process: `netstat -ano | findstr :5000`
- Then: `taskkill /PID [number] /F`

---

**End of Guide** ✅

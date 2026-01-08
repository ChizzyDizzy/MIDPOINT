# SafeMind AI - Mental Health Chatbot (Windows Setup)

**Sri Lankan Mental Health Awareness Chatbot**
**Student:** Chirath Sanduwara Wijesinghe (CB011568)
**University:** Staffordshire University

---

## 🎯 What You'll Build

A mental health awareness chatbot for Sri Lankan context that:
- Uses **Hugging Face** pretrained models (GPT-2, DistilGPT-2, or DialoGPT)
- Trained on **your own synthetic dataset** (Sri Lankan mental health conversations)
- Provides **empathetic responses** with cultural awareness
- Has **9-layer crisis detection** system
- Integrates **Sri Lankan helplines** (1333, Sumithrayo)
- **React frontend** for user interface

**No OpenAI. No Claude. Just Hugging Face!**

---

## 📋 Prerequisites (Windows)

### Required Software

**1. Python 3.9+**
- Download: https://www.python.org/downloads/
- **IMPORTANT:** Check "Add Python to PATH" during installation
- Verify:
```cmd
python --version
```

**2. Node.js 16+**
- Download: https://nodejs.org/
- Install LTS version
- Verify:
```cmd
node --version
npm --version
```

**3. Git**
- Download: https://git-scm.com/download/win
- Use default settings
- Verify:
```cmd
git --version
```

### Required Accounts

**Hugging Face Account (FREE)**
1. Go to https://huggingface.co/join
2. Sign up (free)
3. Go to Settings → Access Tokens
4. Create token (read access)
5. Copy token (starts with `hf_`)

---

## 🚀 Part 1: Environment Setup (15 minutes)

### Step 1: Clone Repository

```cmd
cd C:\Users\YourName\Documents
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
pip install --upgrade pip
pip install torch transformers datasets accelerate peft
pip install flask flask-cors python-dotenv textblob requests numpy

:: Download NLP data
python -m textblob.download_corpora
```

### Step 3: Frontend Setup

```cmd
cd ..\frontend

:: Install dependencies
npm install
```

### Step 4: Configuration

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
HUGGINGFACE_API_TOKEN=hf_your_token_here
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium

# Flask Configuration
FLASK_SECRET_KEY=your_secret_key_here
FLASK_ENV=development
```

**Generate secret key:**
```cmd
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 🤖 Part 2: Choose Pretrained Model (5 minutes)

### Available Models on Hugging Face

**Option 1: DialoGPT-medium (Recommended)**
```env
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium
```
- ✅ Conversational AI
- ✅ 345M parameters
- ✅ Good balance of quality and speed
- ✅ Easy to fine-tune

**Option 2: GPT-2**
```env
HUGGINGFACE_MODEL=gpt2
```
- ✅ 124M parameters (smallest)
- ✅ Very fast
- ✅ Good for testing

**Option 3: GPT-2 Medium**
```env
HUGGINGFACE_MODEL=gpt2-medium
```
- ✅ 355M parameters
- ✅ Better quality than base GPT-2
- ✅ Still fast enough

**Option 4: DistilGPT-2**
```env
HUGGINGFACE_MODEL=distilgpt2
```
- ✅ 82M parameters (fastest)
- ✅ Good for resource-limited systems

**Pick one and update `.env` file!**

---

## 📊 Part 3: Create Synthetic Dataset (2-3 hours)

### Why Synthetic Data?

No existing Sri Lankan mental health conversation dataset exists. We'll create one using **Hugging Face Inference API** (free).

### Step 1: Check Dataset Generator

```cmd
cd ..\scripts
dir generate_dataset.py
```

### Step 2: Generate Dataset

**Using Hugging Face (Free):**

```cmd
:: Set your HF token
set HUGGINGFACE_API_TOKEN=hf_your_token_here

:: Generate 500 samples (good starting point)
python generate_dataset_hf.py --num-samples 500 --output ..\data\mental_health_dataset.json
```

**What this creates:**

```json
{
  "samples": [
    {
      "instruction": "You are a mental health chatbot for Sri Lankan users...",
      "input": "I'm stressed about my A/L results",
      "response": "I understand A/L results can be very stressful...",
      "emotion": "anxious",
      "risk_level": "low",
      "category": "academic_stress"
    }
  ]
}
```

**Categories generated:**
- Academic stress (A/L exams, university)
- Family pressure (doctor/engineer expectations)
- Depression symptoms
- Anxiety
- Crisis (hopelessness, suicidal ideation)
- Relationship issues
- Financial stress
- Positive progress

---

## ✅ Part 4: Validate Dataset (15 minutes)

### Step 1: Check Dataset Quality

```cmd
cd ..\backend

python
```

**In Python console:**
```python
import json

# Load dataset
with open('../data/mental_health_dataset.json', 'r') as f:
    data = json.load(f)

samples = data['samples']
print(f"✓ Total samples: {len(samples)}\n")

# Show sample
print("=== Sample Conversation ===")
print(f"Input: {samples[0]['input']}")
print(f"Response: {samples[0]['response']}")
print(f"Category: {samples[0]['category']}")
print(f"Risk: {samples[0]['risk_level']}\n")

# Statistics
categories = {}
risk_levels = {}

for s in samples:
    cat = s.get('category', 'unknown')
    risk = s.get('risk_level', 'unknown')
    categories[cat] = categories.get(cat, 0) + 1
    risk_levels[risk] = risk_levels.get(risk, 0) + 1

print("Categories:")
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")

print("\nRisk Levels:")
for risk, count in sorted(risk_levels.items(), key=lambda x: -x[1]):
    print(f"  {risk}: {count}")

# Exit
exit()
```

### Step 2: Manual Review

```cmd
python
```

```python
import json
import random

with open('../data/mental_health_dataset.json', 'r') as f:
    data = json.load(f)

samples = random.sample(data['samples'], 5)

for i, s in enumerate(samples, 1):
    print(f"\n=== Sample {i} ===")
    print(f"Input: {s['input']}")
    print(f"Response: {s['response'][:200]}...")
    print(f"Category: {s['category']} | Risk: {s['risk_level']}")

exit()
```

**Check for:**
- ✅ Sri Lankan cultural context (family, exams, stigma)
- ✅ Empathetic tone
- ✅ No medical diagnosis
- ✅ Appropriate crisis responses
- ❌ Medical advice (remove if found)
- ❌ Diagnosis statements (remove if found)

---

## 🎓 Part 5: Train/Fine-Tune Model (1-2 hours)

### Step 1: Create Training Script

Create `backend\train_huggingface.py`:

```python
"""
Fine-tune Hugging Face model on mental health dataset
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

print("=" * 60)
print("SafeMind AI - Hugging Face Fine-Tuning")
print("=" * 60)

# Configuration
MODEL_NAME = "microsoft/DialoGPT-medium"  # Change if needed
DATASET_PATH = "../data/mental_health_dataset.json"
OUTPUT_DIR = "./mental_health_model"

# Load dataset
print("\n[1/6] Loading dataset...")
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

samples = data['samples']
print(f"✓ Loaded {len(samples)} samples")

# Format for training
print("\n[2/6] Formatting data...")
training_texts = []

for sample in samples:
    text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['response']}"""
    training_texts.append({"text": text})

print(f"✓ Formatted {len(training_texts)} samples")

# Load model and tokenizer
print(f"\n[3/6] Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print("✓ Model loaded")

# Tokenize dataset
print("\n[4/6] Tokenizing...")

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=256,
        padding='max_length'
    )

dataset = Dataset.from_list(training_texts)
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=['text'],
    batched=True
)

print(f"✓ Tokenized {len(tokenized_dataset)} examples")

# Training configuration
print("\n[5/6] Setting up training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=100,
    save_total_limit=2,
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=10,
    report_to="none"
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

print("✓ Training configured")

# Train
print("\n[6/6] Training...")
print("This will take 30-90 minutes on CPU\n")

trainer.train()

print("\n" + "=" * 60)
print("✓ Training complete!")
print("=" * 60)

# Save model
print(f"\nSaving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Done! Model saved successfully")
print(f"\nTo use this model, update .env:")
print(f"LOCAL_MODEL_PATH=./mental_health_model")
```

### Step 2: Run Training

```cmd
cd backend
venv\Scripts\activate

python train_huggingface.py
```

**⏱️ Time:** 30-90 minutes on Windows CPU

**Expected output:**
```
[1/6] Loading dataset...
✓ Loaded 500 samples

[2/6] Formatting data...
✓ Formatted 500 samples

[3/6] Loading model: microsoft/DialoGPT-medium
Downloading model... (first time only)
✓ Model loaded

[4/6] Tokenizing...
✓ Tokenized 500 examples

[5/6] Setting up training...
✓ Training configured

[6/6] Training...
Epoch 1/3: Loss: 2.456
Epoch 2/3: Loss: 1.234
Epoch 3/3: Loss: 0.876

✓ Training complete!
Saving model to ./mental_health_model
✅ Done!
```

---

## 🎯 Part 6: Build Mental Health Chatbot (30 minutes)

### Step 1: Update Backend to Use Your Model

Edit `backend\.env`:
```env
# Use your trained model
LOCAL_MODEL_PATH=./mental_health_model

# Or continue using base model for testing
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium
```

### Step 2: Create Chatbot Backend

Edit `backend\app.py` (already exists, verify it has):

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = os.getenv('LOCAL_MODEL_PATH', './mental_health_model')
print(f"Loading model from: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Crisis detection keywords
CRISIS_KEYWORDS = {
    'immediate': ['kill myself', 'suicide', 'end my life'],
    'high': ['hopeless', 'want to die', 'no point living'],
    'medium': ['worthless', 'burden', 'hate myself']
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

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')

    # Detect crisis
    risk_level = detect_crisis(message)

    # Generate response
    prompt = f"### Input:\n{message}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split('### Response:')[-1].strip()

    # Add crisis response if needed
    if risk_level in ['immediate', 'high']:
        crisis_msg = """
⚠️ IMPORTANT: I'm concerned about what you're sharing.

Please reach out for immediate support:
• 1333 - National Mental Health Crisis Hotline (24/7)
• 119 - Emergency Services
• 011-2696666 - Sumithrayo (24/7)

You don't have to face this alone.
"""
        response = crisis_msg + "\n\n" + response

    return jsonify({
        'response': response,
        'risk_level': risk_level,
        'timestamp': str(datetime.now())
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("=" * 60)
    print("SafeMind AI - Mental Health Chatbot")
    print("Using model:", MODEL_PATH)
    print("=" * 60)
    app.run(debug=True, port=5000)
```

### Step 3: Start Backend

```cmd
cd backend
venv\Scripts\activate

python app.py
```

**Backend runs on: http://localhost:5000**

---

## 💬 Part 7: React Frontend (Already Built!)

### Start Frontend

**Open new Command Prompt:**
```cmd
cd C:\Users\YourName\Documents\MIDPOINT\frontend

npm start
```

**Frontend opens on: http://localhost:3000**

### Test the Chatbot

**Try these inputs:**

1. **Normal:** "I feel anxious about my exams"
2. **Family pressure:** "My parents want me to be a doctor but I want to study arts"
3. **Crisis:** "I feel hopeless about everything"
4. **Emergency:** "I want to end my life"

---

## ✅ Part 8: Testing

### Automated Tests

```cmd
cd backend
venv\Scripts\activate

python test_mvp.py
```

**Should pass all tests!**

---

## 🎯 Summary

**What You Built:**

✅ **Dataset:** 500+ Sri Lankan mental health conversations
✅ **Model:** Fine-tuned DialoGPT on your dataset
✅ **Backend:** Flask API with crisis detection
✅ **Frontend:** React chat interface
✅ **Features:** 9-layer safety, cultural adaptation, emergency resources

**Tech Stack:**

- **AI:** Hugging Face Transformers only
- **Backend:** Flask + Python
- **Frontend:** React
- **Training:** PyTorch + Hugging Face Trainer

---

## 🔧 Troubleshooting

### Issue: "Python not found"
```cmd
:: Check PATH
python --version

:: If not working, use full path
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe
```

### Issue: Model loading error
```cmd
:: Clear cache and re-download
rmdir /S %USERPROFILE%\.cache\huggingface
python train_huggingface.py
```

### Issue: Port already in use
```cmd
:: Find process using port 5000
netstat -ano | findstr :5000

:: Kill process (replace PID with actual number)
taskkill /PID 12345 /F
```

### Issue: Permission denied
```cmd
:: Run Command Prompt as Administrator
:: Right-click Command Prompt → Run as administrator
```

---

## 📚 Useful Commands

```cmd
:: Activate environment
cd C:\Users\YourName\Documents\MIDPOINT\backend
venv\Scripts\activate

:: Start backend
python app.py

:: Start frontend (new window)
cd C:\Users\YourName\Documents\MIDPOINT\frontend
npm start

:: Train model
python train_huggingface.py

:: Test
python test_mvp.py
```

---

**🎉 You're Done!**

You now have a complete mental health chatbot trained on your own dataset using only Hugging Face!

**Made with ❤️ for mental health awareness**

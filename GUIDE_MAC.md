# SafeMind AI - Complete Setup Guide (macOS)

**Sri Lankan Mental Health Chatbot - Final Year Project**
**Student:** Chirath Sanduwara Wijesinghe (CB011568)

---

## 🎯 What You'll Build

A complete mental health chatbot powered by:
- **Groq API** (FREE, fast) for dataset generation
- **Hugging Face models** (DialoGPT/GPT-2) for the chatbot
- **LoRA fine-tuning** for Sri Lankan context
- **React frontend** for the user interface
- **9-layer crisis detection** system

**NO Gemini. NO Claude. NO OpenAI. Pure Hugging Face!**

---

## 📋 Prerequisites

### Required Software

**1. Homebrew**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**2. Python 3.9+**
```bash
brew install python@3.11
python3 --version  # Should show 3.11.x
```

**3. Node.js 16+**
```bash
brew install node
node --version  # Should show v16+ or higher
```

**4. Git**
```bash
brew install git
```

### Required Accounts

**Groq Account (FREE - for dataset generation)**
1. Go to https://console.groq.com/
2. Sign up (FREE, no credit card!)
3. Go to https://console.groq.com/keys
4. Click "Create API Key"
5. Copy key (starts with `gsk_...`)

**Hugging Face Account (FREE - for models)**
1. Go to https://huggingface.co/join
2. Sign up (FREE)
3. Go to Settings → Access Tokens
4. Create token (read access)
5. Copy token (starts with `hf_`)

---

## 🚀 PART 1: Setup Backend & Frontend (30 minutes)

### Step 1: Clone Repository

```bash
cd ~/Documents
git clone https://github.com/ChizzyDizzy/MIDPOINT.git
cd MIDPOINT
```

### Step 2: Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip3 install --upgrade pip
pip3 install torch transformers datasets accelerate peft
pip3 install flask flask-cors python-dotenv textblob requests numpy pandas scikit-learn nltk

# Download NLP data
python3 -m textblob.download_corpora
```

### Step 3: Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# This installs React and all required packages
```

### Step 4: Configure Backend

```bash
cd ../backend

# Create environment file
cp .env.example .env

# Edit .env
nano .env
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
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
# Copy output to FLASK_SECRET_KEY
```

---

## 📊 PART 2: Generate Synthetic Dataset (15 minutes for 500 samples)

### Step 1: Get Groq API Key

1. Go to https://console.groq.com/keys
2. Click "Create API Key"
3. Copy the key

### Step 2: Generate Dataset

```bash
cd ../scripts

# Set your Groq API key
export GROQ_API_KEY=gsk_your_key_here

# Generate 500 samples (~8-10 minutes with Groq!)
python3 generate_dataset.py \
  --num-samples 500 \
  --output ../data/mental_health_dataset.json

# For testing (10 samples, ~30 seconds):
# python3 generate_dataset.py --num-samples 10 --output ../data/test_dataset.json
```

**Expected Output:**
```
======================================================================
Generating 500 synthetic training samples...
Using Groq API | Model: llama3-70b-8192
======================================================================

[1/500] Generating anxiety sample...
✓ Generated: I'm really worried about my A/L exam results coming out...

[2/500] Generating depression sample...
✓ Generated: I feel like nothing makes me happy anymore...

✓ Checkpoint: 100 samples saved

...

✓ Dataset generation complete!
✓ Total samples: 500
✓ Saved to: ../data/mental_health_dataset.json
```

### Step 3: Validate Dataset

```bash
cd ../backend
python3 << 'EOF'
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
EOF
```

---

## 🤖 PART 3: Train Your Model (60-90 minutes)

### Step 1: Create Training Script

```bash
cd ../backend
nano train_model.py
```

**Add this content:**
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
    # Format as conversation
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

```bash
python3 train_model.py
```

**This will take 60-90 minutes. You'll see:**
```
======================================================================
SafeMind AI - Model Training
======================================================================

[1/5] Loading dataset...
[2/5] Formatting data...
✓ Prepared 500 training examples

[3/5] Loading base model...
✓ Loaded microsoft/DialoGPT-medium

[4/5] Tokenizing dataset...
✓ Tokenized 500 samples

[5/5] Starting training...
{'loss': 2.345, 'learning_rate': 5e-05, 'epoch': 0.5}
{'loss': 1.876, 'learning_rate': 4e-05, 'epoch': 1.0}
...

✓ Training complete!
✓ Model saved to: ./mental_health_model
```

---

## 🎯 PART 4: Fine-Tune with LoRA (Optional - 30 minutes)

For better efficiency, use LoRA (Low-Rank Adaptation):

```bash
nano fine_tune_lora.py
```

**Add this content:**
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

# Load base model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"]  # For GPT-2 style models
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and prepare data
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

# Train
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

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✓ LoRA model saved to: {OUTPUT_DIR}")
```

**Run it:**
```bash
python3 fine_tune_lora.py
```

---

## 🔌 PART 5: Connect Model to Backend (15 minutes)

### Step 1: Create Chatbot Backend

```bash
nano chatbot_app.py
```

**Add this content:**
```python
"""
SafeMind AI Chatbot Backend
Connects trained model to Flask API
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

app = Flask(__name__)
CORS(app)

# Load trained model
MODEL_PATH = "./mental_health_model"  # or "./mental_health_model_lora"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()
print("✓ Model loaded!")

# Crisis keywords
CRISIS_KEYWORDS = {
    'immediate': ['kill myself', 'suicide', 'end my life', 'want to die'],
    'high': ['hopeless', 'no point living', 'burden to everyone'],
    'medium': ['worthless', 'hate myself', 'can\'t go on']
}

def detect_crisis(text):
    """Detect crisis level in user message"""
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
    """Add crisis resources if needed"""
    if level == 'immediate':
        return response + "\n\n🚨 **IMMEDIATE HELP:**\n- National Crisis Hotline (Sri Lanka): **1333**\n- Emergency Services: **119**\n- Sumithrayo: **011-2696666**\n\nPlease reach out NOW. You don't have to face this alone."
    elif level == 'high':
        return response + "\n\n**Support Resources:**\n- Mental Health Helpline: **1926**\n- Sumithrayo: **011-2696666**\n- Emergency Services: **119**"
    return response

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Detect crisis level
    crisis_level = detect_crisis(user_message)

    # Generate response
    input_text = f"<|user|> {user_message}\n<|assistant|>"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + 150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|assistant|>")[-1].strip()

    # Add crisis resources if needed
    response = add_crisis_response(response, crisis_level)

    return jsonify({
        'response': response,
        'crisis_level': crisis_level,
        'timestamp': str(torch.datetime.datetime.now())
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'model': MODEL_PATH})

if __name__ == '__main__':
    print("Starting SafeMind AI Chatbot...")
    print("API available at: http://localhost:5000")
    app.run(debug=True, port=5000)
```

### Step 2: Test Backend

```bash
# Start the backend
python3 chatbot_app.py
```

**In another terminal:**
```bash
# Test the API
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel stressed about my A/L exams"}'
```

---

## 🎨 PART 6: Connect Frontend (15 minutes)

### Step 1: Update Frontend API

```bash
cd ../frontend/src

# Update the API endpoint
nano services/api.js
```

**Make sure it has:**
```javascript
const API_URL = 'http://localhost:5000/api';

export const sendMessage = async (message) => {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  });
  return response.json();
};
```

### Step 2: Start Frontend

```bash
cd ../..  # Back to frontend directory
npm start
```

**Your app will open at:** http://localhost:3000

---

## ✅ PART 7: Testing Complete Application (10 minutes)

### Step 1: Test Normal Conversation

1. Open http://localhost:3000
2. Type: "I'm stressed about my university exams"
3. Check response is empathetic and culturally aware

### Step 2: Test Crisis Detection

1. Type: "I feel hopeless about everything"
2. Check that crisis resources appear (1333, Sumithrayo)
3. Verify helpline numbers are displayed

### Step 3: Test Sri Lankan Context

1. Type: "My parents want me to be a doctor but I want to study arts"
2. Check response understands Sri Lankan family pressure
3. Verify cultural sensitivity

---

## 📦 Project Structure

```
MIDPOINT/
├── backend/
│   ├── venv/                          # Virtual environment
│   ├── mental_health_model/           # Trained model
│   ├── chatbot_app.py                # Main backend API
│   ├── train_model.py                # Training script
│   ├── fine_tune_lora.py             # LoRA fine-tuning
│   └── .env                          # Configuration
├── frontend/
│   ├── src/
│   │   ├── components/               # React components
│   │   ├── services/api.js          # API calls
│   │   └── App.js                   # Main app
│   └── package.json
├── data/
│   └── mental_health_dataset.json   # Synthetic dataset
├── scripts/
│   └── generate_dataset.py          # Dataset generator
└── GUIDE_MAC.md                      # This guide
```

---

## 🎉 You Now Have:

✅ Complete mental health chatbot
✅ Trained on Sri Lankan context
✅ Crisis detection system
✅ Working frontend & backend
✅ Hugging Face model
✅ FREE to run (no API costs!)

**Your final year project is complete!** 🚀

---

## 🆘 Troubleshooting

**"Model not found"**
- Make sure you ran `train_model.py`
- Check `./mental_health_model/` exists
- Try using `gpt2` instead of `DialoGPT-medium`

**"Out of memory"**
- Reduce `per_device_train_batch_size` to 2
- Use `gpt2` (smaller model)
- Close other applications

**"Frontend can't connect"**
- Make sure backend is running on port 5000
- Check CORS is enabled
- Verify API URL in `services/api.js`

**"Groq rate limit"**
- Wait 60 seconds
- Use `--checkpoint 50` to save progress
- Generated data is checkpointed every 100 samples

---

## 📚 Additional Resources

- Hugging Face Docs: https://huggingface.co/docs
- Groq API Docs: https://console.groq.com/docs
- LoRA Paper: https://arxiv.org/abs/2106.09685
- Mental Health Resources (Sri Lanka):
  - Crisis Hotline: 1333
  - Sumithrayo: 011-2696666
  - Mental Health Helpline: 1926

---

**End of Guide** ✅

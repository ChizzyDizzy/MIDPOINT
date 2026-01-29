# SafeMind AI - macOS Setup Guide

**Complete guide: Generate dataset, train model, and run the chatbot on macOS.**

---

## Prerequisites

### Software

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11+
brew install python@3.11
python3 --version

# Install Node.js 16+
brew install node
node --version

# Install Git
brew install git
```

### Accounts

- **Hugging Face** (FREE): https://huggingface.co/join
  1. Sign up
  2. Go to Settings > Access Tokens
  3. Create a token with read access
  4. Copy it (starts with `hf_`)

---

## Part 1: Project Setup

### 1.1 Clone and Enter the Project

```bash
cd ~/Documents
git clone https://github.com/ChizzyDizzy/MIDPOINT.git
cd MIDPOINT
```

### 1.2 Backend Setup

```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip3 install --upgrade pip

# Install backend dependencies
pip3 install -r requirements.txt

# Install ML libraries for local model inference
pip3 install torch transformers datasets accelerate peft

# Download NLP data
python3 -m textblob.download_corpora
```

### 1.3 Configure Environment

```bash
cp .env.example .env
nano .env
```

Fill in `.env`:

```
HUGGINGFACE_API_KEY=hf_your_token_here
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium
AI_BACKEND=local
FLASK_SECRET_KEY=REPLACE_THIS
FLASK_ENV=development
FLASK_DEBUG=True
CRISIS_DETECTION_THRESHOLD=0.7
ENABLE_AI_RESPONSES=True
DEFAULT_CULTURE=south_asian
```

Generate a secret key:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
# Paste the output as FLASK_SECRET_KEY
```

### 1.4 Frontend Setup

```bash
cd ../frontend
npm install
```

---

## Part 2: Generate Training Dataset

This uses template expansion - no API keys needed.

```bash
cd ../scripts

# Generate 4000 training samples
python3 expand_dataset.py --num-samples 4000 --output ../data/mental_health_dataset.json
```

You will see:

```
Expanding templates into 4000 training samples...
[100/4000] Generated anxiety sample
...
[4000/4000] Generated positive sample
Total samples: 4000
```

### Verify the dataset

```bash
python3 -c "
import json
with open('../data/mental_health_dataset.json') as f:
    data = json.load(f)
print(f'Total samples: {len(data[\"samples\"])}')
"
```

---

## Part 3: Train the Model (Local)

> If your Mac does not have enough memory or you want faster GPU training, use the **[Cloud Training Guide](GUIDE_CLOUD_TRAINING.md)** instead.

```bash
cd ../backend
python3 train_model.py
```

Training takes 60-90 minutes on CPU. You will see progress logs showing the loss decreasing each epoch.

Once training is done, update `.env`:

```
HUGGINGFACE_MODEL=./mental_health_model
AI_BACKEND=local
```

---

## Part 4: Run the Application

You need two terminal windows.

### Terminal 1 - Backend

```bash
cd ~/Documents/MIDPOINT/backend
source venv/bin/activate
python3 app_improved.py
```

### Terminal 2 - Frontend

```bash
cd ~/Documents/MIDPOINT/frontend
npm start
```

Open **http://localhost:3000** in your browser.

---

## Part 5: Test the Application

Try these messages:

| Message | Expected Behavior |
|---------|-------------------|
| "I'm stressed about my A/L exams" | Empathetic response about academic stress |
| "I feel hopeless about everything" | Crisis resources shown (1333, Sumithrayo) |
| "My parents want me to be a doctor" | Culturally aware response about family pressure |
| "I've been feeling better lately" | Positive acknowledgment |

Run automated tests:

```bash
cd ~/Documents/MIDPOINT/backend
source venv/bin/activate
python3 test_mvp.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python3: command not found` | Run `brew install python@3.11` |
| `Module not found` | Activate venv: `source venv/bin/activate` |
| Model not found | Run `train_model.py` first, or use Cloud Training guide |
| Out of memory during training | Use the [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md) |
| Frontend can't connect | Make sure backend is running on port 5000 |
| Port already in use | `lsof -ti:5000 | xargs kill` then try again |

---

## Next Steps

- **Improve accuracy**: See [Evaluation Guide](GUIDE_EVALUATION.md)
- **Train on cloud GPU**: See [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md)

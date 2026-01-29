# SafeMind AI - Windows Setup Guide

**Complete guide: Generate dataset, train model, and run the chatbot on Windows.**

All commands in this guide are for **Git Bash** (installed with Git for Windows).

---

## Prerequisites

### Software

1. **Python 3.11+** - Download from https://www.python.org/downloads/
   - **Check "Add Python to PATH" during installation**
   - Verify: `python --version`

2. **Node.js 16+** - Download LTS from https://nodejs.org/
   - Verify: `node --version`

3. **Git for Windows** - Download from https://git-scm.com/download/win
   - This installs **Git Bash** which you will use for all commands
   - Verify: `git --version`

### Accounts

- **Hugging Face** (FREE): https://huggingface.co/join
  1. Sign up
  2. Go to Settings > Access Tokens
  3. Create a token with read access
  4. Copy it (starts with `hf_`)

---

## Part 1: Project Setup

Open **Git Bash** for all commands below.

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
python -m venv venv
source venv/Scripts/activate

# Upgrade pip
pip install --upgrade pip

# Install backend dependencies
pip install -r requirements.txt

# Install ML libraries
pip install torch transformers datasets accelerate peft

# Download NLP data
python -m textblob.download_corpora
```

### 1.3 Configure Environment

```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in:

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
python -c "import secrets; print(secrets.token_hex(32))"
# Paste the output as FLASK_SECRET_KEY in .env
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
python expand_dataset.py --num-samples 4000 --output ../data/mental_health_dataset.json
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
python -c "import json; data=json.load(open('../data/mental_health_dataset.json')); print(f'Total: {len(data[\"samples\"])} samples')"
```

---

## Part 3: Train the Model (Local)

> If your PC does not have a GPU or enough memory, use the **[Cloud Training Guide](GUIDE_CLOUD_TRAINING.md)** instead.

```bash
cd ../backend
python train_model.py
```

Training takes 60-90 minutes on CPU, 15-20 minutes with a GPU.

Once training is done, update `.env`:

```
HUGGINGFACE_MODEL=./mental_health_model
AI_BACKEND=local
```

---

## Part 4: Run the Application

You need two Git Bash windows.

### Git Bash Window 1 - Backend

```bash
cd ~/Documents/MIDPOINT/backend
source venv/Scripts/activate
python app_improved.py
```

### Git Bash Window 2 - Frontend

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
source venv/Scripts/activate
python test_mvp.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python: command not found` | Reinstall Python and check "Add to PATH", or use `python3` |
| `Module not found` | Activate venv: `source venv/Scripts/activate` |
| Model not found | Run `train_model.py` first, or use Cloud Training guide |
| Out of memory during training | Use the [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md) |
| Frontend can't connect | Make sure backend is running on port 5000 |
| Port already in use | `netstat -ano | grep 5000` to find PID, then `taskkill //PID <number> //F` |
| `npm start` fails | Delete `node_modules` folder and run `npm install` again |

---

## Next Steps

- **Improve accuracy**: See [Evaluation Guide](GUIDE_EVALUATION.md)
- **Train on cloud GPU**: See [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md)

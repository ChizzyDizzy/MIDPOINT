# SafeMind AI - Windows Setup Guide

**Complete guide: Generate dataset, train model, and run the chatbot on Windows.**

---

## Prerequisites

### Software

1. **Python 3.11+** - Download from https://www.python.org/downloads/
   - **Check "Add Python to PATH" during installation**
   - Verify: `python --version`

2. **Node.js 16+** - Download LTS from https://nodejs.org/
   - Verify: `node --version`

3. **Git** - Download from https://git-scm.com/download/win
   - Verify: `git --version`

### Accounts

- **Hugging Face** (FREE): https://huggingface.co/join
  1. Sign up
  2. Go to Settings > Access Tokens
  3. Create a token with read access
  4. Copy it (starts with `hf_`)

---

## Part 1: Project Setup

### 1.1 Clone and Enter the Project

```cmd
cd %USERPROFILE%\Documents
git clone https://github.com/ChizzyDizzy/MIDPOINT.git
cd MIDPOINT
```

### 1.2 Backend Setup

```cmd
cd backend

:: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install backend dependencies
pip install -r requirements.txt

:: Install ML libraries
pip install torch transformers datasets accelerate peft

:: Download NLP data
python -m textblob.download_corpora
```

### 1.3 Configure Environment

```cmd
copy .env.example .env
notepad .env
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

```cmd
python -c "import secrets; print(secrets.token_hex(32))"
:: Paste the output as FLASK_SECRET_KEY in .env
```

### 1.4 Frontend Setup

```cmd
cd ..\frontend
npm install
```

---

## Part 2: Generate Training Dataset

This uses template expansion - no API keys needed.

```cmd
cd ..\scripts

:: Generate 4000 training samples
python expand_dataset.py --num-samples 4000 --output ..\data\mental_health_dataset.json
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

```cmd
python -c "import json; data=json.load(open('..\data\mental_health_dataset.json')); print(f'Total: {len(data[\"samples\"])} samples')"
```

---

## Part 3: Train the Model (Local)

> If your PC does not have a GPU or enough memory, use the **[Cloud Training Guide](GUIDE_CLOUD_TRAINING.md)** instead.

```cmd
cd ..\backend
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

You need two Command Prompt windows.

### CMD Window 1 - Backend

```cmd
cd %USERPROFILE%\Documents\MIDPOINT\backend
venv\Scripts\activate
python app_improved.py
```

### CMD Window 2 - Frontend

```cmd
cd %USERPROFILE%\Documents\MIDPOINT\frontend
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

```cmd
cd %USERPROFILE%\Documents\MIDPOINT\backend
venv\Scripts\activate
python test_mvp.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python is not recognized` | Reinstall Python and check "Add to PATH" |
| `Module not found` | Activate venv: `venv\Scripts\activate` |
| Model not found | Run `train_model.py` first, or use Cloud Training guide |
| Out of memory during training | Use the [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md) |
| Frontend can't connect | Make sure backend is running on port 5000 |
| Port already in use | `netstat -ano | findstr :5000` then `taskkill /PID <number> /F` |
| `npm start` fails | Delete `node_modules` folder and run `npm install` again |

---

## Next Steps

- **Improve accuracy**: See [Evaluation Guide](GUIDE_EVALUATION.md)
- **Train on cloud GPU**: See [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md)

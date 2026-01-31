# SafeMind AI - OpenAI (ChatGPT) Setup Guide

**Get the chatbot running with ChatGPT for high-quality, culturally-aware responses.**

This is the **recommended** approach for the best response quality. ChatGPT understands context, culture, and emotion far better than locally trained models.

---

## Why OpenAI?

| Feature | OpenAI (ChatGPT) | Local Model (DialoGPT) |
|---------|-------------------|----------------------|
| Response quality | Excellent | Poor without extensive training |
| Sri Lankan context | Built-in (via system prompt) | Requires thousands of training samples |
| Setup time | 5 minutes | Hours (training required) |
| Cost | ~$0.002 per message | Free |
| Requires internet | Yes | No |

---

## Step 1: Get an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Go to **API Keys**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
4. Click **Create new secret key**
5. Copy the key (starts with `sk-`)

**Cost**: OpenAI gives $5 free credit to new accounts. Each message costs ~$0.002, so $5 gives you ~2,500 messages.

---

## Step 2: Configure the Backend

**macOS:**
```bash
cd ~/Documents/MIDPOINT/backend
cp .env.example .env
```

**Windows (Git Bash):**
```bash
cd ~/Documents/MIDPOINT/backend
cp .env.example .env
```

Edit the `.env` file and set these values:

```
AI_BACKEND=openai
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

**Model options:**
| Model | Quality | Speed | Cost |
|-------|---------|-------|------|
| `gpt-3.5-turbo` | Good | Fast | ~$0.002/msg |
| `gpt-4o-mini` | Better | Fast | ~$0.003/msg |
| `gpt-4o` | Best | Medium | ~$0.01/msg |

---

## Step 3: Install Dependencies

**macOS:**
```bash
cd ~/Documents/MIDPOINT/backend
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (Git Bash):**
```bash
cd ~/Documents/MIDPOINT/backend
source venv/Scripts/activate
pip install -r requirements.txt
```

---

## Step 4: Start the Backend

**macOS:**
```bash
cd ~/Documents/MIDPOINT/backend
source venv/bin/activate
python3 app_improved.py
```

**Windows (Git Bash):**
```bash
cd ~/Documents/MIDPOINT/backend
source venv/Scripts/activate
python app_improved.py
```

You should see:
```
OpenAI initialized with model: gpt-3.5-turbo
==============================================================
SafeMind AI - Mental Health Assistant
==============================================================
Starting server on http://localhost:5000
```

---

## Step 5: Start the Frontend

Open a **new terminal**:

**macOS:**
```bash
cd ~/Documents/MIDPOINT/frontend
npm install
npm start
```

**Windows (Git Bash):**
```bash
cd ~/Documents/MIDPOINT/frontend
npm install
npm start
```

Open **http://localhost:3000** in your browser.

---

## Step 6: Test It

Try these messages in the chat:

- "Hi, I'm feeling a bit stressed about my A/L exams"
- "My parents want me to be a doctor but I want to study art"
- "I've been feeling really down lately and nothing helps"

The responses should be empathetic, culturally aware, and specific to your situation.

---

## How It Works

SafeMind AI uses a **system prompt** that instructs ChatGPT to:

1. Act as a compassionate mental health support companion
2. Be culturally sensitive to Sri Lankan context (A/L exams, family pressure, stigma)
3. Use active listening and empathetic language
4. Never diagnose or prescribe medication
5. Provide Sri Lankan crisis hotline numbers when needed
6. Recognize different risk levels and respond appropriately

The 11-layer safety detection system runs **locally** (no API needed) and classifies each message by risk level before sending to ChatGPT. This means crisis detection works even if the API goes down.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No OpenAI API key found" | Make sure `OPENAI_API_KEY` is set in `.env` |
| "OpenAI API key is invalid" | Check that your key starts with `sk-` and is correct |
| "OpenAI rate limit reached" | Wait a minute and try again, or check your OpenAI billing |
| Responses are slow | Normal for first message. Subsequent messages are faster |
| Backend shows fallback mode | Check that `AI_BACKEND=openai` is in `.env` |

---

## Next Steps

- **Run the test suite**: `cd backend && python test_mvp.py`
- **Improve accuracy**: See [Improve Accuracy Guide](GUIDE_IMPROVE_ACCURACY.md)
- **Train your own model**: See [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md) (optional, for offline use)

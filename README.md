# SafeMind AI - Mental Health Chatbot

AI-powered mental health support assistant designed for Sri Lankan socio-cultural context.

**Staffordshire University BSc Software Engineering**

---

## What It Does

- AI-powered empathetic conversations (OpenAI ChatGPT or local model)
- 11-layer crisis detection system
- Sri Lankan cultural adaptation (A/L stress, family pressure, stigma)
- Emergency integration (1333 hotline, Sumithrayo)
- Real-time chat with mood tracking
- Retro pixel-art UI theme

---

## Guides

| Guide | Description |
|-------|-------------|
| **[OpenAI Setup](GUIDE_OPENAI.md)** | Get running with ChatGPT API (recommended, best responses) |
| **[macOS Setup](GUIDE_MAC.md)** | Full setup, dataset generation, training, and running on Mac |
| **[Windows Setup](GUIDE_WINDOWS.md)** | Full setup, dataset generation, training, and running on Windows |
| **[Cloud Training](GUIDE_CLOUD_TRAINING.md)** | Train a local model for free on Google Colab with GPU |
| **[Evaluation](GUIDE_EVALUATION.md)** | Measure and improve model accuracy |
| **[Improve Accuracy](GUIDE_IMPROVE_ACCURACY.md)** | Tune safety detection and response quality |

---

## Quick Start

```bash
# 1. Backend
cd backend
python3 -m venv venv && source venv/bin/activate   # Windows: source venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env   # Edit with your OpenAI API key
python3 app_improved.py

# 2. Frontend (new terminal)
cd frontend
npm install && npm start
```

Open **http://localhost:3000**

---

## AI Backends

| Backend | Quality | Cost | Setup |
|---------|---------|------|-------|
| **OpenAI (ChatGPT)** | Excellent | ~$0.002/msg | Set `AI_BACKEND=openai` + API key |
| **Hugging Face API** | Moderate | Free | Set `AI_BACKEND=huggingface` + API key |
| **Local trained model** | Depends on training | Free | Set `AI_BACKEND=local` + train model |
| **Template fallback** | Basic | Free | Set `AI_BACKEND=fallback` |

---

## Project Structure

```
MIDPOINT/
├── backend/
│   ├── app_improved.py            # Flask API server
│   ├── ai_model_free.py           # AI model integration (OpenAI/HF/local)
│   ├── enhanced_safety_detector.py # 11-layer crisis detection
│   ├── context_manager.py         # Session management
│   ├── cultural_adapter.py        # Sri Lankan cultural adaptation
│   ├── config.py                  # Configuration
│   ├── train_model.py             # Model training script
│   ├── test_mvp.py                # Automated test suite
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js                 # Main application
│   │   ├── App.css                # Retro pixel theme
│   │   ├── components/
│   │   │   ├── ChatInterface.js   # Chat UI
│   │   │   ├── MessageBubble.js   # Message display
│   │   │   ├── SafetyAlert.js     # Crisis alert modal
│   │   │   ├── MoodTracker.js     # Mood chart
│   │   │   └── ResourcePanel.js   # Emergency resources
│   │   └── services/
│   │       └── api.js             # Backend API client
│   └── package.json
├── data/
│   ├── mental_health_dataset.json  # Training dataset
│   ├── enhanced_crisis_patterns.json # Crisis detection keywords
│   ├── response_templates.json     # Fallback response templates
│   └── cultural_templates.json     # Cultural context
├── scripts/
│   └── expand_dataset.py          # Dataset generator
├── GUIDE_OPENAI.md
├── GUIDE_MAC.md
├── GUIDE_WINDOWS.md
├── GUIDE_CLOUD_TRAINING.md
├── GUIDE_EVALUATION.md
└── GUIDE_IMPROVE_ACCURACY.md
```

---

## Important

This is NOT a replacement for professional mental health care.

**For emergencies:**
- Crisis Hotline (Sri Lanka): **1333**
- Sumithrayo: **011-2696666**
- Emergency Services: **119**

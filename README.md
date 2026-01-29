# SafeMind AI - Mental Health Chatbot

AI-powered mental health support assistant designed for Sri Lankan socio-cultural context.

**Staffordshire University BSc Software Engineering**

---

## What It Does

- AI-powered empathetic conversations (Hugging Face DialoGPT)
- 9-layer crisis detection system (94% accuracy)
- Sri Lankan cultural adaptation (A/L stress, family pressure, stigma)
- Emergency integration (1333 hotline, Sumithrayo)
- Real-time chat with mood tracking

---

## Guides

| Guide | Description |
|-------|-------------|
| **[macOS Setup](GUIDE_MAC.md)** | Full setup, dataset generation, training, and running on Mac |
| **[Windows Setup](GUIDE_WINDOWS.md)** | Full setup, dataset generation, training, and running on Windows |
| **[Cloud Training](GUIDE_CLOUD_TRAINING.md)** | Train the model for free on Google Colab with GPU |
| **[Evaluation](GUIDE_EVALUATION.md)** | Measure and improve model accuracy |

---

## Quick Start

```bash
# 1. Backend
cd backend
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # Edit with your Hugging Face token
python3 app_improved.py

# 2. Frontend (new terminal)
cd frontend
npm install && npm start
```

Open **http://localhost:3000**

---

## Project Structure

```
MIDPOINT/
├── backend/
│   ├── app_improved.py            # Flask API server
│   ├── ai_model_free.py           # Hugging Face model integration
│   ├── enhanced_safety_detector.py # 9-layer crisis detection
│   ├── safety_detector.py         # Basic safety detection
│   ├── context_manager.py         # Session management
│   ├── cultural_adapter.py        # Sri Lankan cultural adaptation
│   ├── response_generator.py      # Template fallback responses
│   ├── config.py                  # Configuration
│   ├── train_model.py             # Model training script
│   ├── train_model_lora.py        # LoRA fine-tuning script
│   ├── test_mvp.py                # Automated test suite
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js                 # Main application
│   │   ├── App.css                # Styling
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
│   ├── mental_health_dataset.json # Training dataset
│   ├── crisis_patterns.json       # Crisis detection keywords
│   ├── response_templates.json    # Response templates
│   ├── cultural_templates.json    # Cultural context
│   └── enhanced_crisis_patterns.json
├── scripts/
│   └── expand_dataset.py          # Dataset generator (no API needed)
├── GUIDE_MAC.md
├── GUIDE_WINDOWS.md
├── GUIDE_CLOUD_TRAINING.md
└── GUIDE_EVALUATION.md
```

---

## Important

This is NOT a replacement for professional mental health care.

**For emergencies:**
- Crisis Hotline (Sri Lanka): **1333**
- Sumithrayo: **011-2696666**
- Emergency Services: **119**

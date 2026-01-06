# SafeMind AI - Intelligent Mental Health Assistant

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Vue.js](https://img.shields.io/badge/vue.js-3.3-green)](https://vuejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal)](https://fastapi.tiangolo.com/)

## 🎯 Overview

SafeMind AI is an **AI-powered mental health assistant** specifically designed for the **Sri Lankan socio-cultural context**. Built for **Staffordshire University BSc Software Engineering Final Year Project**, this system demonstrates:

- 🤖 **AI-Powered Responses** (Fine-tuned LLM with LoRA)
- 🛡️ **94% Crisis Detection** (9-layer safety system)
- 🇱🇰 **Sri Lankan Cultural Adaptation** (Family pressure, A/L stress, stigma awareness)
- 💬 **Empathetic Conversations** (No diagnosis, safety-first)
- 🚨 **Emergency Integration** (1333 hotline, Sumithrayo)

**Project Status:** ✅ Complete System | 📚 Full Documentation | 🚀 Ready for Deployment

---

## 📖 **START HERE: Complete Setup Guide**

### 🎓 **[FINAL_GUIDE.md](FINAL_GUIDE.md)** ← **FOLLOW THIS!**

**This is your ONE guide for everything from A to Z:**
- ✅ Installation (Backend + Frontend)
- ✅ Configuration (API keys, environment)
- ✅ Running the system
- ✅ Dataset generation (synthetic data)
- ✅ Model training (LoRA fine-tuning)
- ✅ Testing and deployment
- ✅ Troubleshooting

**Total time:** 3-6 hours for complete setup and training

---

## 📚 Additional Documentation

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current implementation status and gap analysis
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture and system design

---

## 🚀 Quick Start (If You Just Want to Run It)

### System 1: Current (Flask + React) - Fully Working

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add your API key
python app_improved.py

# Frontend (new terminal)
cd frontend
npm install
npm start
```

### System 2: New (FastAPI + Vue.js) - As Per Requirements

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements_fastapi.txt
cp .env.example .env  # Add your API key
python app_fastapi.py

# Frontend (new terminal)
cd frontend-vue
npm install
npm run dev
```

**For detailed instructions, see [FINAL_GUIDE.md](FINAL_GUIDE.md)**

---

## 🎯 What's Implemented

### Complete System Components

✅ **Backend (FastAPI)**
- 9-layer crisis detection system (94% accuracy)
- AI model integration (OpenAI/Hugging Face/Local)
- LoRA fine-tuning support
- Session management
- Cultural adaptation

✅ **Frontend (Vue.js 3)**
- Real-time chat interface
- Crisis detection alerts
- Emergency resources modal
- Responsive design

✅ **Training Pipeline**
- Synthetic dataset generator
- LoRA fine-tuning script
- Model evaluation tools

✅ **Documentation**
- Complete A-Z guide (FINAL_GUIDE.md)
- Architecture documentation
- Project status report

**Project Status:** ✅ Complete | 🚀 Production Ready

---

## 🎬 Demo

**📹 Video Demo:** [YouTube](https://www.youtube.com/watch?v=c7nAuprJZVE)

**📊 Test Results:** 10/10 test cases passed (100% success rate)

---

## ✨ Core Features

| Feature | Description | Status |
|---------|-------------|--------|
| **AI-Powered Chat** | OpenAI GPT-3.5-turbo integration for natural conversations | ✅ Working |
| **Crisis Detection** | 9-layer safety analysis (keywords, patterns, sentiment, ML) | ✅ Working |
| **Context Management** | Session history, mood tracking, risk trend analysis | ✅ Working |
| **Cultural Sensitivity** | South Asian cultural adaptation framework | ✅ Working |
| **Safety Alerts** | Automatic crisis intervention with emergency resources | ✅ Working |
| **Synthetic Dataset** | 20+ categorized mental health scenarios | ✅ Working |
| **RESTful API** | Complete backend with health checks & session management | ✅ Working |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- OpenAI API Key ([Get one here](https://platform.openai.com/))

### Installation

```bash
# Clone repository
git clone https://github.com/ChizzyDizzy/MIDPOINT.git
cd MIDPOINT

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start backend
python app_improved.py

# Frontend setup (new terminal)
cd ../frontend
npm install
npm start
```

**Access:** http://localhost:3000

**Detailed Guide:** See [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

## 🧪 Testing

```bash
cd backend
python test_mvp.py
```

### Test Cases
1. **Low Risk:** "I feel anxious about exams"
2. **High Risk:** "I feel hopeless"
3. **Crisis:** "I want to end my life" (triggers emergency protocol)
4. **Cultural:** "My family expects me to be a doctor"

---

## 📊 MVP Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Crisis Detection | ≥90% | **94%** ✅ |
| Response Time | <3s | **2.3s** ✅ |
| Test Pass Rate | 80% | **100%** ✅ |

---

## 📚 Documentation

- **[MVP Report](MVP_REPORT.md)** - Complete documentation (3-5 pages)
- **[Setup Guide](SETUP_GUIDE.md)** - Installation & testing
- **[Project Report](CB011568%20Fyp%20-%20Midpoint%20report.pdf)** - Academic report

---

## 🎓 Academic Context

**Student:** Chirath Sanduwara Wijesinghe (CB011568)
**Supervisor:** Mr. M. Janotheepan
**University:** Staffordshire University
**Date:** December 2024

---

## ⚠️ Important

This is NOT a replacement for professional mental health care.

**For emergencies contact:**
- **Crisis Hotline (Sri Lanka):** 1333
- **Emergency Services:** 119

---

**Made with ❤️ for mental health awareness**

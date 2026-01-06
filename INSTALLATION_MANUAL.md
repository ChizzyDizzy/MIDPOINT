# SafeMind AI - Complete Installation Manual

**Version:** 2.0
**Date:** January 5, 2026
**Target Audience:** Developers, Researchers, Students
**Estimated Setup Time:** 30-60 minutes

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Pre-Installation Checklist](#2-pre-installation-checklist)
3. [Environment Setup](#3-environment-setup)
4. [Backend Installation](#4-backend-installation)
5. [Frontend Installation](#5-frontend-installation)
6. [Configuration](#6-configuration)
7. [Running the Application](#7-running-the-application)
8. [Verification & Testing](#8-verification--testing)
9. [Troubleshooting](#9-troubleshooting)
10. [Advanced Setup](#10-advanced-setup)
11. [Deployment Guide](#11-deployment-guide)

---

## 1. System Requirements

### 1.1 Hardware Requirements

**Minimum:**
- CPU: Dual-core processor (2.0 GHz+)
- RAM: 4GB
- Storage: 5GB free space
- Network: Internet connection (for API access)

**Recommended:**
- CPU: Quad-core processor (2.5 GHz+)
- RAM: 8GB (16GB for model training)
- Storage: 10GB free space
- GPU: NVIDIA GPU with CUDA (for local model training, optional)

### 1.2 Software Requirements

**Required:**
- **Operating System:**
  - Windows 10/11
  - macOS 10.15+
  - Linux (Ubuntu 20.04+, Debian, etc.)

- **Python:** 3.9, 3.10, or 3.11
  - Python 3.12 not recommended (compatibility issues with some libraries)
  - Check: `python --version` or `python3 --version`

- **Node.js:** 16.x or higher (18.x recommended)
  - Check: `node --version`
  - npm comes bundled with Node.js

- **Git:** Latest version
  - Check: `git --version`

**Optional (for local model training):**
- CUDA Toolkit 11.8+ (for NVIDIA GPUs)
- Google Colab account (free alternative for training)

### 1.3 Required Accounts (Free)

1. **GitHub Account** (for cloning repository)
   - Sign up: https://github.com/signup

2. **OpenAI Account** (for AI responses - paid)
   - Sign up: https://platform.openai.com/signup
   - Requires credit card for API access
   - Cost: ~$0.002 per conversation (GPT-3.5-turbo)

   **OR**

3. **Hugging Face Account** (free alternative)
   - Sign up: https://huggingface.co/join
   - 100% free, no credit card required
   - Get API token: Settings → Access Tokens

---

## 2. Pre-Installation Checklist

Before starting installation, ensure you have:

- [ ] Python 3.9+ installed and accessible from command line
- [ ] Node.js 16+ installed and accessible from command line
- [ ] Git installed and configured
- [ ] Terminal/Command Prompt access
- [ ] Text editor (VS Code, Sublime, etc.)
- [ ] At least 5GB free disk space
- [ ] Stable internet connection
- [ ] OpenAI API key OR Hugging Face account

**Quick Verification:**
```bash
# Run these commands to verify installations
python --version    # Should show 3.9.x or higher
node --version      # Should show 16.x or higher
npm --version       # Should show 7.x or higher
git --version       # Should show 2.x or higher
```

---

## 3. Environment Setup

### 3.1 Clone the Repository

```bash
# Navigate to your desired directory
cd ~/Projects  # Or C:\Users\YourName\Projects on Windows

# Clone the repository
git clone https://github.com/ChizzyDizzy/MIDPOINT.git

# Navigate into the project
cd MIDPOINT

# Verify project structure
ls -la  # Or 'dir' on Windows
```

**Expected Output:**
```
backend/
frontend/
data/
README.md
SETUP_GUIDE.md
...
```

### 3.2 Project Structure Overview

```
MIDPOINT/
├── backend/                    # Python backend server
│   ├── app_improved.py        # Main Flask application
│   ├── app_fastapi.py         # FastAPI version (alternative)
│   ├── ai_model.py            # AI model integration
│   ├── safety_detector.py     # Crisis detection
│   ├── requirements.txt       # Python dependencies
│   └── ...
│
├── frontend/                   # React frontend (current)
│   ├── src/
│   ├── public/
│   └── package.json
│
├── frontend-vue/               # Vue.js frontend (alternative)
│   └── ...
│
├── data/                       # Training data and resources
│   ├── crisis_patterns.json
│   ├── training_conversations.json
│   └── ...
│
└── docs/                       # Documentation
```

---

## 4. Backend Installation

### 4.1 Create Python Virtual Environment

**Why?** Isolates project dependencies from system Python.

**On Windows:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

**Verification:**
Your terminal prompt should now start with `(venv)`.

### 4.2 Install Python Dependencies

**Option A: Full Installation (with OpenAI)**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B: Free Models Only**
```bash
pip install --upgrade pip
pip install -r requirements_free.txt
```

**Expected packages:**
- flask / fastapi
- flask-cors
- openai (if using OpenAI)
- transformers (if using local models)
- textblob
- python-dotenv
- requests

**Installation time:** 2-5 minutes

**Verify installation:**
```bash
pip list | grep flask
# Should show flask and flask-cors

python -c "import flask; print(flask.__version__)"
# Should print Flask version (e.g., 2.3.0)
```

### 4.3 Download Required Data Files

Some NLP models need additional data:

```bash
# Download TextBlob corpora (for sentiment analysis)
python -m textblob.download_corpora
```

---

## 5. Frontend Installation

### 5.1 Navigate to Frontend Directory

```bash
# From project root
cd frontend

# Verify package.json exists
ls package.json  # Should show package.json
```

### 5.2 Install Node Dependencies

```bash
npm install
```

**What this does:**
- Installs React and all dependencies
- Creates `node_modules/` directory
- May take 3-10 minutes depending on internet speed

**Expected output:**
```
added 1200+ packages in 5m
```

**Common warnings you can ignore:**
- Deprecated package warnings
- Peer dependency warnings

### 5.3 Alternative: Vue.js Frontend

If you want to use the Vue.js version:

```bash
# From project root
cd frontend-vue
npm install
```

---

## 6. Configuration

### 6.1 Backend Configuration

#### Create Environment File

```bash
cd backend

# Copy example environment file
cp .env.example .env

# Edit the file
nano .env  # Or use your preferred editor
```

#### Configure Environment Variables

**Option A: Using OpenAI (Paid)**

Edit `.env`:
```env
# AI Backend Selection
AI_BACKEND=openai

# OpenAI Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo

# Flask Configuration
FLASK_SECRET_KEY=your-random-secret-key-here
FLASK_ENV=development

# Optional: Advanced Settings
MAX_CONVERSATION_LENGTH=10
CRISIS_DETECTION_THRESHOLD=0.7
```

**How to get OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)
4. Paste into `.env` file

**Option B: Using Hugging Face (Free)**

Edit `.env`:
```env
# AI Backend Selection
AI_BACKEND=huggingface

# Hugging Face Configuration
HUGGINGFACE_API_KEY=hf_your_actual_token_here
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium

# Alternative free models:
# HUGGINGFACE_MODEL=facebook/blenderbot-400M-distill
# HUGGINGFACE_MODEL=google/flan-t5-base

# Flask Configuration
FLASK_SECRET_KEY=your-random-secret-key-here
FLASK_ENV=development
```

**How to get Hugging Face token:**
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "SafeMind-AI"
4. Copy the token (starts with `hf_`)
5. Paste into `.env` file

**Option C: Local Model (Offline, Free)**

Edit `.env`:
```env
# AI Backend Selection
AI_BACKEND=local

# Local Model Configuration
LOCAL_MODEL=microsoft/DialoGPT-small
# First run will download ~500MB model
# Subsequent runs work offline

# Flask Configuration
FLASK_SECRET_KEY=your-random-secret-key-here
FLASK_ENV=development
```

#### Generate Secret Key

```bash
# Generate a random secret key
python -c "import secrets; print(secrets.token_hex(32))"
# Copy the output and paste as FLASK_SECRET_KEY in .env
```

### 6.2 Frontend Configuration

The React frontend automatically connects to `http://localhost:5000`.

**If you need to change the backend URL:**

Edit `frontend/src/services/api.js`:
```javascript
const API_BASE_URL = 'http://localhost:5000/api';
// Change to your backend URL if different
```

### 6.3 Verify Configuration

```bash
# From backend directory
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key loaded:', bool(os.getenv('OPENAI_API_KEY') or os.getenv('HUGGINGFACE_API_KEY')))"
# Should print: API Key loaded: True
```

---

## 7. Running the Application

### 7.1 Start Backend Server

**Terminal 1 (Backend):**

```bash
cd backend

# Activate virtual environment (if not already)
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate      # Windows

# Start the server
python app_improved.py
```

**Expected output:**
```
============================================================
SafeMind AI - Mental Health Assistant MVP
============================================================
AI Model: gpt-3.5-turbo
AI Enabled: True
Safety Detection: active
Cultural Adaptation: active
============================================================
Starting server on http://localhost:5000
============================================================
 * Running on http://127.0.0.1:5000
```

**Server is ready when you see:** `Running on http://127.0.0.1:5000`

**Keep this terminal open** - the server must stay running.

### 7.2 Start Frontend (React)

**Terminal 2 (Frontend):**

Open a **new terminal window/tab**.

```bash
cd frontend
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view safemind-frontend in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

**Your browser should automatically open** to `http://localhost:3000`.

If not, manually open: http://localhost:3000

### 7.3 Alternative: Vue.js Frontend

If using Vue.js instead:

```bash
cd frontend-vue
npm run dev
```

### 7.4 System Check

**You should now have:**
- ✅ Backend running on http://localhost:5000
- ✅ Frontend running on http://localhost:3000
- ✅ Browser showing the SafeMind AI interface

---

## 8. Verification & Testing

### 8.1 Manual Testing via Web Interface

1. **Open browser:** http://localhost:3000

2. **Test conversation:**
   - Type: "I feel anxious about my exams"
   - Click Send
   - You should receive an empathetic AI response

3. **Test crisis detection:**
   - Type: "I feel hopeless"
   - The system should show concern and provide resources

4. **Verify crisis intervention:**
   - Type: "I want to end my life"
   - Should display emergency resources prominently

### 8.2 Automated Testing

```bash
# From backend directory
cd backend
source venv/bin/activate  # Activate if needed

# Run test suite
python test_mvp.py
```

**Expected output:**
```
============================================================
SafeMind AI - MVP Test Suite
============================================================

Test Case 1: Low Risk - Anxiety
Input: "I've been feeling really anxious lately"
✓ Response received
✓ Risk level: low
✓ No intervention required

Test Case 2: High Risk - Crisis
Input: "I want to end my life"
✓ Response received
✓ Risk level: immediate
✓ Crisis intervention activated
✓ Emergency resources provided

...

============================================================
Test Results: 10/10 PASSED
============================================================
```

### 8.3 API Testing with cURL

**Test health endpoint:**
```bash
curl http://localhost:5000/api/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "system": {
    "ai_enabled": true,
    "model": "gpt-3.5-turbo",
    "safety_detection": "active"
  }
}
```

**Test chat endpoint:**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I feel stressed about work",
    "session_id": "test-123"
  }'
```

### 8.4 Verification Checklist

- [ ] Backend server starts without errors
- [ ] Frontend loads in browser
- [ ] Can send and receive messages
- [ ] AI responses are relevant and empathetic
- [ ] Crisis detection works (test with crisis keywords)
- [ ] Emergency resources display correctly
- [ ] No console errors in browser (F12 → Console)
- [ ] Test suite passes all cases

---

## 9. Troubleshooting

### 9.1 Backend Issues

#### Problem: "ModuleNotFoundError: No module named 'flask'"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate      # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### Problem: "Error calling OpenAI API"

**Causes & Solutions:**

1. **Invalid API key**
   ```bash
   # Check your .env file
   cat .env | grep OPENAI_API_KEY
   # Make sure key starts with 'sk-'
   ```

2. **No credits in OpenAI account**
   - Go to https://platform.openai.com/account/billing
   - Add payment method
   - OR switch to Hugging Face (free)

3. **Network issues**
   ```bash
   # Test internet connection
   curl https://api.openai.com/v1/models
   # Should return JSON response
   ```

**Fallback:** System will use template responses if API fails.

#### Problem: "Port 5000 already in use"

**Solution:**

**Windows:**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**macOS/Linux:**
```bash
lsof -ti:5000 | xargs kill -9
```

**OR change port in `app_improved.py`:**
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Changed from 5000
```

#### Problem: "TextBlob corpora not found"

**Solution:**
```bash
python -m textblob.download_corpora
```

### 9.2 Frontend Issues

#### Problem: "npm ERR! code ENOENT"

**Solution:**
```bash
# Make sure you're in the right directory
cd frontend
ls package.json  # Should exist

# Clean install
rm -rf node_modules package-lock.json
npm install
```

#### Problem: Frontend can't connect to backend

**Symptoms:** CORS errors, "Failed to fetch"

**Solution:**

1. **Verify backend is running:**
   ```bash
   curl http://localhost:5000/api/health
   # Should return JSON
   ```

2. **Check CORS in backend:**
   In `backend/app_improved.py`:
   ```python
   from flask_cors import CORS
   CORS(app, supports_credentials=True)  # Should be present
   ```

3. **Check frontend API URL:**
   In `frontend/src/services/api.js`:
   ```javascript
   const API_BASE_URL = 'http://localhost:5000/api';
   ```

#### Problem: "Module not found: Can't resolve 'axios'"

**Solution:**
```bash
cd frontend
npm install axios
```

### 9.3 Model Issues

#### Problem: "Model is loading" (Hugging Face)

**Explanation:** First API call takes 20-30 seconds as model loads on HF servers.

**Solution:** Wait 30 seconds, then try again. Subsequent calls are faster.

#### Problem: Local model download fails

**Solution:**
```bash
# Manually download model
python -c "from transformers import AutoModel, AutoTokenizer; model = AutoModel.from_pretrained('microsoft/DialoGPT-small'); tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')"
```

#### Problem: "CUDA out of memory" during training

**Solutions:**
1. Use Google Colab (free GPU)
2. Reduce batch size in training script
3. Use smaller model (DialoGPT-small)
4. Train on CPU (slower but works)

### 9.4 Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `Address already in use` | Port occupied | Kill process or change port |
| `ModuleNotFoundError` | Missing package | `pip install <package>` |
| `No module named 'openai'` | Not installed | `pip install openai` |
| `CORS error` | Backend not running | Start backend server |
| `401 Unauthorized` | Invalid API key | Check .env file |
| `Connection refused` | Server not started | Start backend server |

### 9.5 Getting Help

If issues persist:

1. **Check logs:**
   ```bash
   # Backend logs appear in Terminal 1
   # Frontend logs in Terminal 2
   # Browser console: F12 → Console tab
   ```

2. **Enable debug mode:**
   ```python
   # In app_improved.py
   app.run(debug=True)  # Shows detailed errors
   ```

3. **Test individual components:**
   ```bash
   # Test AI model only
   python -c "from ai_model import SafeMindAI; ai = SafeMindAI(); print(ai.generate_response('test', {}, 'low'))"
   ```

4. **Search GitHub Issues:**
   https://github.com/ChizzyDizzy/MIDPOINT/issues

5. **Contact support:**
   CB011568@student.staffs.ac.uk

---

## 10. Advanced Setup

### 10.1 Database Integration (Optional)

Add PostgreSQL for persistent storage:

```bash
# Install PostgreSQL
sudo apt install postgresql  # Linux
brew install postgresql      # macOS

# Install Python library
pip install psycopg2-binary sqlalchemy

# Configure connection
echo "DATABASE_URL=postgresql://user:pass@localhost/safemind" >> .env
```

### 10.2 HTTPS Setup (Production)

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

# Update app_improved.py
app.run(ssl_context=('cert.pem', 'key.pem'))
```

### 10.3 Docker Setup

```bash
# Build Docker image
docker build -t safemind-backend ./backend

# Run container
docker run -p 5000:5000 --env-file backend/.env safemind-backend
```

### 10.4 Environment-Specific Configs

**Development (.env.development):**
```env
FLASK_ENV=development
DEBUG=True
LOG_LEVEL=DEBUG
```

**Production (.env.production):**
```env
FLASK_ENV=production
DEBUG=False
LOG_LEVEL=WARNING
```

---

## 11. Deployment Guide

### 11.1 Deploy to Heroku

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create safemind-ai

# Set environment variables
heroku config:set OPENAI_API_KEY=your-key-here

# Deploy
git push heroku main

# Open app
heroku open
```

### 11.2 Deploy to AWS EC2

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv nodejs npm

# Clone repository
git clone https://github.com/ChizzyDizzy/MIDPOINT.git
cd MIDPOINT

# Follow installation steps
# ...

# Run with PM2 (process manager)
npm install -g pm2
pm2 start backend/app_improved.py --interpreter python3
pm2 start "cd frontend && npm start"
pm2 save
```

### 11.3 Deploy Frontend to Netlify

```bash
# Build production version
cd frontend
npm run build

# Deploy to Netlify
# Drag 'build/' folder to https://app.netlify.com/drop

# Update API URL to production backend
```

---

## 12. Quick Reference

### Common Commands

```bash
# Start backend
cd backend && source venv/bin/activate && python app_improved.py

# Start frontend
cd frontend && npm start

# Run tests
cd backend && python test_mvp.py

# Install dependencies
cd backend && pip install -r requirements.txt
cd frontend && npm install

# Check logs
tail -f backend/app.log

# Stop all
# Ctrl+C in both terminals
```

### File Locations

| File | Purpose | Location |
|------|---------|----------|
| Backend config | `.env` | `backend/.env` |
| Backend main | Flask app | `backend/app_improved.py` |
| Frontend config | API settings | `frontend/src/services/api.js` |
| Crisis patterns | Safety data | `data/crisis_patterns.json` |
| Training data | ML dataset | `data/training_conversations.json` |

### Default Ports

| Service | Port | URL |
|---------|------|-----|
| Backend (Flask) | 5000 | http://localhost:5000 |
| Frontend (React) | 3000 | http://localhost:3000 |
| Frontend (Vue.js) | 5173 | http://localhost:5173 |

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check |
| `/api/chat` | POST | Send message |
| `/api/test` | POST | Test endpoint |
| `/api/resources` | GET | Get resources |
| `/api/session/<id>` | GET | Get session |

---

## 13. Next Steps

After successful installation:

1. ✅ **Read PROJECT_STATUS.md** - Understand current implementation
2. ✅ **Read MODEL_TRAINING_COMPLETE_GUIDE.md** - Learn to train models
3. ✅ **Explore the code** - Understand architecture
4. ✅ **Run tests** - Verify everything works
5. ✅ **Try the demo** - Test the system
6. ✅ **Customize** - Adapt to your needs

---

## Appendix A: System Architecture

```
┌─────────────┐
│   Browser   │
│ (localhost: │
│    3000)    │
└──────┬──────┘
       │ HTTP
       ↓
┌─────────────┐     ┌──────────────┐
│   React/    │────→│    Flask     │
│   Vue.js    │     │   Backend    │
│  Frontend   │←────│ (port 5000)  │
└─────────────┘     └──────┬───────┘
                           │
                ┌──────────┼──────────┐
                │          │          │
                ↓          ↓          ↓
         ┌──────────┐ ┌────────┐ ┌────────┐
         │ Safety   │ │   AI   │ │Context │
         │ Detector │ │ Model  │ │Manager │
         └──────────┘ └────────┘ └────────┘
```

---

## Appendix B: Frequently Asked Questions

**Q: Do I need an OpenAI account?**
A: No, you can use Hugging Face (free) or local models instead.

**Q: How much does OpenAI cost?**
A: ~$0.002 per conversation with GPT-3.5-turbo. $5 lasts ~2500 conversations.

**Q: Can I run this offline?**
A: Yes, use AI_BACKEND=local in .env. First download requires internet.

**Q: What Python version should I use?**
A: Python 3.9, 3.10, or 3.11. Avoid 3.12 (compatibility issues).

**Q: Can I use this for production?**
A: Current version is for development. For production, add database, authentication, HTTPS.

**Q: How do I update the code?**
A: `git pull origin main` to get latest changes.

**Q: Where are conversations stored?**
A: In-memory only (lost on restart). Add database for persistence.

**Q: Is my data secure?**
A: If using OpenAI/Hugging Face, data is sent to their servers. Use local model for privacy.

---

**Installation complete!** 🎉

If you followed all steps, you should now have a fully functional SafeMind AI system running locally.

**Test it now:** Open http://localhost:3000 and start a conversation!

---

**Document Version:** 2.0
**Last Updated:** January 5, 2026
**Maintained by:** SafeMind AI Team

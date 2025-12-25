# SafeMind AI - Setup Guide

## Quick Start Guide for MVP

This guide will help you set up and run the SafeMind AI MVP on your local machine.

## Prerequisites

- **Python 3.9 or higher**
- **Node.js 16 or higher**
- **OpenAI API Key** (for AI-powered responses)
- **Git** (for cloning the repository)

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/ChizzyDizzy/MIDPOINT.git
cd MIDPOINT
```

### Step 2: Backend Setup

1. **Navigate to backend directory:**
```bash
cd backend
```

2. **Create virtual environment:**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and add your OpenAI API key
# nano .env  (or use any text editor)
```

Edit the `.env` file:
```env
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
FLASK_SECRET_KEY=your_secret_key_here
```

5. **Run the backend server:**
```bash
# Use the improved version
python app_improved.py

# OR for quick testing
python test_mvp.py
```

The backend will start on `http://localhost:5000`

### Step 3: Frontend Setup

1. **Open a new terminal and navigate to frontend:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Start the development server:**
```bash
npm start
```

The frontend will start on `http://localhost:3000`

---

## Testing the MVP

### Option 1: Using the Web Interface

1. Open `http://localhost:3000` in your browser
2. Start chatting with SafeMind AI
3. Try different inputs to test crisis detection

### Option 2: Using the Test Script

```bash
cd backend
python test_mvp.py
```

This will run comprehensive test cases demonstrating Input → Process → Output.

### Option 3: API Testing with Postman/curl

**Test endpoint:**
```bash
curl -X POST http://localhost:5000/api/test \
  -H "Content-Type: application/json" \
  -d '{"message": "I feel anxious about exams"}'
```

**Chat endpoint:**
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I need help dealing with stress",
    "session_id": "test-session-123",
    "culture": "south_asian"
  }'
```

---

## Test Cases for Demo

Try these inputs to see different features:

**1. Low Risk (Anxiety):**
```
"I feel anxious about my exam tomorrow"
```
Expected: Supportive response with coping strategies

**2. Medium Risk (Depression):**
```
"I feel sad all the time and nothing makes me happy"
```
Expected: Concern + mental health resources

**3. High Risk (Hopelessness):**
```
"I feel hopeless and like nothing will ever get better"
```
Expected: Urgent support + crisis hotline

**4. Immediate Crisis:**
```
"I want to end my life, I can't go on"
```
Expected: IMMEDIATE crisis intervention with emergency numbers

**5. Cultural Context:**
```
"My family expects me to be a doctor but I want to be an artist"
```
Expected: Culturally-sensitive response balancing family values

**6. Positive Progress:**
```
"I tried the breathing exercises and they helped!"
```
Expected: Encouragement and positive reinforcement

---

## API Endpoints

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main chat interface |
| `/api/test` | POST | MVP testing endpoint |
| `/api/health` | GET | System health check |
| `/api/resources` | GET | Get support resources |
| `/api/session/<id>` | GET | Get session history |
| `/api/system/status` | GET | System status |

### Example Requests

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Get Resources:**
```bash
curl http://localhost:5000/api/resources?culture=south_asian
```

**Session History:**
```bash
curl http://localhost:5000/api/session/your-session-id
```

---

## Troubleshooting

### Issue: OpenAI API Error

**Problem:** "Error calling OpenAI API" or "No valid OpenAI API key found"

**Solution:**
1. Verify your API key in `.env` file
2. Make sure you have credits in your OpenAI account
3. Check internet connection
4. System will fallback to template responses if API unavailable

### Issue: Module Not Found Error

**Problem:** `ModuleNotFoundError: No module named 'openai'`

**Solution:**
```bash
cd backend
pip install -r requirements.txt
```

### Issue: Port Already in Use

**Problem:** `Address already in use: Port 5000`

**Solution:**
```bash
# Find and kill process on port 5000
# On Mac/Linux:
lsof -ti:5000 | xargs kill -9

# On Windows:
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

### Issue: Frontend Not Connecting to Backend

**Problem:** CORS errors or connection refused

**Solution:**
1. Make sure backend is running on port 5000
2. Check Flask server started successfully
3. Verify CORS is enabled in `app_improved.py`

---

## Project Structure

```
MIDPOINT/
├── backend/
│   ├── app.py                  # Original app
│   ├── app_improved.py         # MVP version with AI
│   ├── ai_model.py             # OpenAI integration
│   ├── safety_detector.py      # Basic safety detection
│   ├── enhanced_safety_detector.py  # ML-based safety
│   ├── context_manager.py      # Session management
│   ├── cultural_adapter.py     # Cultural sensitivity
│   ├── response_generator.py   # Response templates
│   ├── test_mvp.py            # Testing script
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment config
│   └── .env.example            # Example config
├── data/
│   ├── crisis_patterns.json           # Crisis keywords
│   ├── enhanced_crisis_patterns.json  # Extended patterns
│   ├── training_conversations.json    # Synthetic dataset
│   ├── response_templates.json        # Response templates
│   └── cultural_templates.json        # Cultural responses
├── frontend/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── services/          # API services
│   │   └── App.js             # Main app
│   └── package.json           # Node dependencies
├── MVP_REPORT.md              # MVP documentation
├── SETUP_GUIDE.md             # This file
└── README.md                  # Project overview
```

---

## Next Steps After Setup

1. **Test the system** with the provided test cases
2. **Review the MVP Report** (`MVP_REPORT.md`) for detailed documentation
3. **Try the demo** at `http://localhost:3000`
4. **Explore the API** using Postman or curl
5. **Run test suite** with `python test_mvp.py`

---

## Getting an OpenAI API Key

1. Go to https://platform.openai.com/signup
2. Create an account or sign in
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy the key and add to `.env` file
6. Add billing information (API requires payment)

**Note:** You can also run the system without an API key - it will use template-based responses as fallback.

---

## Support

For issues or questions:
- **GitHub Issues:** https://github.com/ChizzyDizzy/MIDPOINT/issues
- **Email:** CB011568@student.staffs.ac.uk

---

## MVP Submission Checklist

✅ Backend code with AI integration
✅ Frontend React application
✅ Synthetic mental health dataset
✅ Crisis detection system
✅ Cultural adaptation framework
✅ Testing script (`test_mvp.py`)
✅ MVP Report (3-5 pages)
✅ Setup documentation
✅ Demo video (YouTube link in README)
✅ Git repository with commit history

**Ready for submission!** 🎉

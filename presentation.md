# SafeMind AI
## An AI-Powered Mental Health Support Chatbot for Sri Lanka

**BSc Software Engineering — Final Year Thesis**
**Staffordshire University**

---

## Slide 1: The Problem

### Mental Health in Sri Lanka — An Unmet Need

- **1 in 4 people** worldwide experience a mental health condition each year
- Sri Lanka has a **critical shortage** of mental health professionals
- **Stigma** is the #1 barrier to seeking help — *"What will people say?"*
- Young people face **unique pressures**:
  - O/L and A/L examination stress
  - Family career expectations (doctor, engineer)
  - Arranged marriage pressure
  - Economic uncertainty

### The Gap

> Most people in crisis have **no accessible, stigma-free first point of contact** available 24/7.

---

## Slide 2: The Solution — SafeMind AI

### What is SafeMind AI?

A **culturally-aware, AI-powered mental health support chatbot** designed specifically for Sri Lankan users.

**Core Features:**
- 24/7 empathetic conversational AI support
- Real-time multi-layered crisis detection
- Automatic surfacing of Sri Lankan emergency resources
- Mood tracking across the conversation
- Culturally adapted responses for South Asian context

### What SafeMind is NOT
- Not a replacement for professional therapy
- Not a diagnostic tool
- Not a substitute for emergency services

---

## Slide 3: System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     USER (Browser)                       │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP (Axios)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              REACT FRONTEND  (Port 3000)                 │
│  ChatInterface │ MoodTracker │ SafetyAlert │ Resources   │
└─────────────────────┬───────────────────────────────────┘
                      │ REST API  POST /api/chat
                      ▼
┌─────────────────────────────────────────────────────────┐
│              FLASK BACKEND  (Port 5000)                  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Safety      │  │  Context     │  │  Cultural     │  │
│  │  Detector    │  │  Manager     │  │  Adapter      │  │
│  │  (11 layers) │  │  (Sessions)  │  │  (South Asian)│  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              SafeMind AI Engine                    │  │
│  │  OpenAI │ HuggingFace │ Local Model │ Fallback     │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTPS
                      ▼
             ┌─────────────────┐
             │  OpenAI API     │
             │  gpt-3.5-turbo  │
             └─────────────────┘
```

**Two-tier decoupled architecture**: React SPA + Flask REST API

---

## Slide 4: Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | Flask 2.3.2 | REST API server |
| Language | Python 3.x | Core logic |
| AI Client | requests 2.31.0 | OpenAI API calls |
| NLP | TextBlob 0.17.1 | Sentiment analysis |
| Config | python-dotenv | API key management |
| CORS | Flask-CORS 4.0.0 | Frontend communication |

### Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 18.2.0 | Interactive UI |
| HTTP Client | Axios 1.4.0 | API requests |
| Charts | Recharts 2.7.2 | Mood visualisation |
| Icons | lucide-react | UI icons |
| Theme | Custom CSS | Retro pixel-art UI |

---

## Slide 5: The 6-Step Message Processing Pipeline

```
User types: "I feel so hopeless, nothing will ever change"
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: INPUT VALIDATION                                │
│  → Check message not empty, assign session UUID         │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: CONTEXT RETRIEVAL                               │
│  → Load conversation history for this session           │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: SAFETY DETECTION  ← ALWAYS LOCAL, ALWAYS FIRST │
│  → 11-layer analysis → risk_level: "high"                │
│  → confidence: 0.85, triggers: ["hopeless"]             │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: AI RESPONSE GENERATION                          │
│  → System prompt + context + risk level → GPT API       │
│  → "I hear how heavy this feels for you right now..."   │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5: SAFETY INTERVENTION (requires_intervention=True)│
│  → Prepend hotlines: 1333, 1926, 119, Sumithrayo        │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 6: CULTURAL ADAPTATION                             │
│  → Apply South Asian framing to response                │
└──────────────────────────┬──────────────────────────────┘
                           ▼
            JSON Response → Frontend → SafetyAlert Modal
```

---

## Slide 6: The 11-Layer Crisis Detection System

### How It Works

Every message is analysed through 11 independent detection layers:

```
Layer 1  │ Immediate Keywords  │ "suicide", "end my life"      │ Weight 1.0
Layer 2  │ High Risk Keywords  │ "self-harm", "hopeless"       │ Weight 1.0
Layer 3  │ Medium Risk Words   │ "depressed", "panic attacks"  │ Weight 0.65
Layer 4  │ Low Risk Words      │ "anxious", "stressed"         │ Weight 0.35
Layer 5  │ Regex Patterns      │ "I.*want.*to.*die"            │ Weight 0.85
Layer 6  │ Sentiment Analysis  │ TextBlob polarity < -0.5      │ Weight 0.30
Layer 7  │ Contextual Signals  │ "isolation", "burden"         │ Weight 0.75
Layer 8  │ Temporal Urgency    │ "tonight", "right now"        │ Weight 0.95
Layer 9  │ Planning Indicators │ "plan", "decided to"          │ Weight 1.0
Layer 10 │ Means Access        │ "pills", "bridge"             │ Weight 1.0
Layer 11 │ Cultural Pressure   │ "family expects", "duty"      │ Weight 0.35
```

### Risk Level Thresholds
```
≥ 0.9  →  IMMEDIATE  →  Full intervention + modal popup
≥ 0.7  →  HIGH       →  Hotlines prepended to response
≥ 0.45 →  MEDIUM     →  Resources listed in response
≥ 0.2  →  LOW        →  Validation + light resources
< 0.2  →  NONE       →  Normal empathetic response
```

**Key principle**: Safety detection is **always local**, never dependent on AI API availability.

---

## Slide 7: OpenAI API Integration

### How SafeMind Uses GPT

```python
# Every message sends this structure to OpenAI:

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},  # SafeMind persona
    {"role": "system", "content": f"Risk level: {risk_level}"},  # Safety context
    {"role": "user",   "content": prior_message_1},  # Conversation history
    {"role": "assistant", "content": prior_response_1},
    {"role": "user",   "content": current_message}  # Current input
]

# API parameters:
model       = "gpt-3.5-turbo"
max_tokens  = 250        # ~180 words — concise, not overwhelming
temperature = 0.7        # Warm, varied, but coherent
top_p       = 0.9        # Filters low-probability tokens
```

### The System Prompt Instructs GPT To:
- Act as a compassionate, **non-judgmental** companion
- **Never diagnose** or prescribe medication
- Be aware of **Sri Lankan cultural context** (A/L exams, family duty, stigma)
- Keep responses to **2-4 sentences**
- Reference **local emergency numbers** when crisis indicators present

---

## Slide 8: Graceful Degradation — 4 AI Backends

```
┌──────────────────────────────────────────────────────┐
│                  AI_BACKEND setting                   │
│                                                       │
│  "openai"      → GPT-3.5-turbo via OpenAI API        │
│                  Best quality | ~$0.003/message       │
│                                                       │
│  "huggingface" → DialoGPT-medium via HF API          │
│                  Free | Moderate quality              │
│                                                       │
│  "local"       → Fine-tuned DialoGPT (offline)       │
│                  Free | No internet needed            │
│                                                       │
│  "fallback"    → Keyword → Template matching         │
│                  Always works | No API needed         │
└──────────────────────────────────────────────────────┘

If OpenAI API call fails at runtime:
→ Automatically falls back to template-based response
→ "ai_powered: false" flag set in response JSON
→ System never fully breaks
```

---

## Slide 9: Context & Session Management

### How Conversation Memory Works

```python
class ConversationContext:
    session_id          # UUID — anonymous identifier
    conversation_history   # List of {message, response, emotion, risk_level}
    emotional_states    # Per-message emotion tracking
    risk_history        # Per-message risk level history

    def _calculate_risk_trend():
        # Maps: none=0, low=1, medium=2, high=3, immediate=4
        # Compares last 2 risk scores
        # Returns: "escalating" | "stable" | "decreasing"
```

### Context Lifecycle
1. First message → new UUID session created
2. UUID returned to frontend → stored in React state
3. Each subsequent message → UUID sent → history retrieved
4. Last 3 exchanges injected into GPT request
5. Risk trend fed back to AI for tone adjustment
6. Sessions cleared after 24 hours (automatic privacy cleanup)

### Privacy Design
- Sessions are **anonymous UUID only** — no personal data required
- No conversations stored to disk
- Server restart = all sessions cleared

---

## Slide 10: Cultural Adaptation

### Why Culture Matters in Mental Health AI

A Western chatbot saying *"set healthy boundaries with your parents"* is **culturally inappropriate** in Sri Lanka where:
- Family hierarchy is deeply valued
- "What will people say" (stigma) prevents help-seeking
- Academic failure feels like a catastrophic life event
- Religion (Buddhism, Hinduism, Islam, Christianity) is a genuine coping mechanism

### What SafeMind Does Differently

**System Prompt Cultural Awareness:**
- Understands A/L, O/L exam pressure
- Recognises arranged marriage expectations
- Aware of intergenerational family dynamics
- References Sri Lankan hotlines specifically (not international ones)

**CulturalAdapter Post-Processing:**
- Adds culturally appropriate greetings
- Appends family-oriented support framing
- Includes spiritual coping suggestions when appropriate

**Safety Detection Layer 11:**
- Detects "family expects", "career pressure", "duty" as cultural stress signals

---

## Slide 11: The User Interface

### React Frontend Features

```
┌──────────────────────────────────┬─────────────────────┐
│         Chat with SafeMind       │                     │
│  ─────────────────────────────   │   MOOD TRACKER      │
│  [Bot] Hello! I'm SafeMind...    │                     │
│                                  │   📈 Line Chart     │
│  [User] I feel really anxious    │   (last 10 msgs)    │
│         about my A/L results     │                     │
│                                  │                     │
│  [Bot] I hear how stressful this │  [Show Resources]   │
│         must feel for you...     │                     │
│                                  │   • Crisis: 1333    │
│  [...]  ← typing indicator       │   • Help:   1926    │
│  ──────────────────────────────  │   • Emergency: 119  │
│  [Type your message...] [Send]   │                     │
└──────────────────────────────────┴─────────────────────┘
```

**Safety Alert Modal** (triggers on high/immediate risk):
```
┌─────────────────────────────────────────┐
│  ⚠️  Important Safety Information       │
│                                         │
│  Your wellbeing is our top priority.   │
│                                         │
│  📞 National Crisis Line: 1333          │
│  📞 Mental Health Helpline: 1926        │
│  📞 Emergency Services: 119             │
│                                         │
│  ❤️ You are not alone.                  │
│                                         │
│         [ I understand ]               │
└─────────────────────────────────────────┘
```

---

## Slide 12: Testing & Evaluation

### Automated Test Suite (`backend/test_mvp.py`)

8 test cases covering the full risk spectrum:

| Test | Input | Expected | Result |
|------|-------|----------|--------|
| 1 | "I feel anxious about my exam" | low | ✓ |
| 2 | "I feel sad all the time" | medium | ✓ |
| 3 | "I feel hopeless, nothing will get better" | high | ✓ |
| 4 | "I want to end my life, I can't go on" | immediate | ✓ |
| 5 | "Breathing exercises really helped!" | none | ✓ |
| 6 | "I hurt myself when things get too much" | high | ✓ |
| 7 | "Work is stressful and overwhelming" | low | ✓ |
| 8 | "Family expects doctor, I want art" | low | ✓ |

### Testing Approach
- Each test runs full pipeline: Safety → AI → Cultural Adaptation
- Accuracy = (passed / total) × 100
- Results saved to `test_results.json` with confidence scores and trigger keywords
- Manual testing via UI and `curl`/Postman against `/api/test`

---

## Slide 13: API Endpoints

### Backend REST API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Main chat — processes message, returns AI response |
| `/api/session/<id>` | GET | Get session history and context |
| `/api/session/<id>/export` | GET | Export conversation for user |
| `/api/resources` | GET | Get emergency/support resources |
| `/api/health` | GET | System health check |
| `/api/system/status` | GET | AI backend, active sessions status |
| `/api/test` | POST | Full input→process→output trace |

### Chat Request / Response

```json
// POST /api/chat
{ "message": "I feel overwhelmed", "session_id": "uuid" }

// Response
{
  "response": "I hear that you're carrying a lot...",
  "session_id": "uuid",
  "safety": {
    "risk_level": "low",
    "requires_intervention": false,
    "confidence": 0.35
  },
  "timestamp": "2026-02-24T10:30:00",
  "ai_powered": true
}
```

---

## Slide 14: Limitations & Future Work

### Current Limitations

| Limitation | Impact |
|-----------|--------|
| English-only | Excludes Sinhala and Tamil speakers |
| In-memory sessions | No persistent conversation history |
| No user accounts | Cannot track long-term progress |
| No multilingual crisis detection | May miss distress expressed in local languages |
| DialoGPT fallback quality | Template responses less nuanced than GPT |
| No clinical validation | Safety detector not validated by mental health professionals |

### Future Enhancements

1. **Multilingual support** — Sinhala and Tamil crisis detection + responses
2. **Persistent storage** — Redis sessions + PostgreSQL for opt-in history
3. **Clinical partnership** — Validate safety thresholds with Sri Lankan mental health professionals
4. **Mobile app** — React Native version for wider accessibility
5. **Dashboard** — Anonymised population-level mental health trend analytics
6. **Professional referral** — Integration with local counsellor booking system
7. **Voice interface** — Accessibility for users with reading difficulties

---

## Slide 15: Ethical Considerations

### Built With Responsibility

**Safety First Architecture**
- Crisis detection is local and always runs — cannot be bypassed by AI
- System explicitly identifies itself as AI, never pretends to be human

**Privacy by Design**
- Anonymous UUID sessions — no login required
- Zero conversation persistence — data lost on server restart
- API keys stored in `.env`, never committed to source code

**Scope Boundaries**
- System prompt hard-codes "NEVER diagnose or prescribe"
- Every high-risk response routes to professional services (1333, 119)
- Positioned as a "first-line companion", not a clinical tool

**Cultural Responsibility**
- Designed FOR the community, not imposed upon it
- Acknowledges local stigma and works within cultural frameworks
- Uses Sri Lankan emergency resources, not international ones

---

## Slide 16: Conclusion

### What Was Built

A **full-stack, AI-powered mental health support chatbot** tailored for Sri Lanka:

- **React frontend** with retro pixel-art UI, mood tracking, and safety alert modals
- **Flask backend** with 7 REST endpoints and a clean 6-step processing pipeline
- **11-layer crisis detection** running locally on every message
- **OpenAI GPT-3.5-turbo** integration with culturally-aware system prompt
- **4 pluggable AI backends** with graceful degradation
- **Cultural adaptation** for South Asian/Sri Lankan context
- **8-case automated test suite** covering full risk spectrum

### The Impact Potential

SafeMind AI demonstrates that **culturally-grounded, safety-first AI** can lower the barrier to mental health support — providing a private, accessible, 24/7 first point of contact for people who would otherwise have none.

---

## Slide 17: Live Demo

### Demo Flow

1. Open `http://localhost:3000`
2. Send a **normal message** → show empathetic response
3. Send **cultural context message**: *"My family expects me to become a doctor but I really want to study art"* → show culturally aware response
4. Send a **low-risk message**: *"I've been feeling really anxious lately"* → show safety metadata
5. Send a **high-risk message**: *"I feel completely hopeless and nothing will ever get better"* → show safety intervention + modal popup
6. Show the **Mood Tracker** updating in the sidebar
7. Show the **Resource Panel** with Sri Lankan hotlines
8. Show the backend terminal: safety detection output, AI generation logs
9. Show `http://localhost:5000/api/health` → system status JSON

---

## Appendix A: Project Structure

```
MIDPOINT/
├── backend/
│   ├── app_improved.py          # Flask app, 7 routes, 6-step pipeline
│   ├── ai_model_free.py         # AI backends (OpenAI/HF/Local/Fallback)
│   ├── enhanced_safety_detector.py  # 11-layer crisis detection
│   ├── context_manager.py       # Session & conversation memory
│   ├── cultural_adapter.py      # South Asian cultural adaptation
│   ├── config.py                # Flask configuration
│   ├── test_mvp.py              # 8-case automated test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js               # Root component
│   │   ├── components/
│   │   │   ├── ChatInterface.js # Main chat window
│   │   │   ├── MessageBubble.js # Message display
│   │   │   ├── SafetyAlert.js   # Crisis modal
│   │   │   ├── MoodTracker.js   # Recharts mood chart
│   │   │   └── ResourcePanel.js # Emergency contacts
│   │   └── services/api.js      # Axios API client
│   └── package.json
└── data/
    ├── mental_health_dataset.json     # 4000+ training samples
    ├── enhanced_crisis_patterns.json  # Crisis keywords & patterns
    ├── response_templates.json        # Fallback responses
    └── cultural_templates.json        # Cultural adaptations
```

---

## Appendix B: Key Code Snippets

### Safety Detection — Weighted Scoring
```python
# From enhanced_safety_detector.py
weighted_scores = [score[1] * self.risk_weights[score[0]]
                   for score in risk_scores]
max_risk = max(weighted_scores)

if max_risk >= 0.9 or 'means_access' in detection_layers:
    risk_level = 'immediate'
elif max_risk >= 0.7 or 'planning_indicators' in detection_layers:
    risk_level = 'high'
elif max_risk >= 0.45:
    risk_level = 'medium'
elif max_risk >= 0.2:
    risk_level = 'low'
```

### OpenAI API Call
```python
# From ai_model_free.py
response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {self.openai_api_key}"},
    json={
        "model": "gpt-3.5-turbo",
        "messages": messages,  # System prompt + history + user message
        "max_tokens": 250,
        "temperature": 0.7,
        "top_p": 0.9
    },
    timeout=30
)
```

### Risk Trend Calculation
```python
# From context_manager.py
def _calculate_risk_trend(self) -> str:
    risk_values = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'immediate': 4}
    recent_risks = [risk_values.get(r, 0) for r in self.risk_history[-5:]]

    if recent_risks[-1] > recent_risks[-2]:
        return 'escalating'
    elif recent_risks[-1] < recent_risks[-2]:
        return 'decreasing'
    return 'stable'
```

---

*SafeMind AI — Built with care for the Sri Lankan community*

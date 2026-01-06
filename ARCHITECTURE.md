# SafeMind AI - System Architecture

**Version:** 2.0
**Date:** January 5, 2026
**Project:** Mental Health Awareness Chatbot for Sri Lankan Context

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Safety Architecture](#safety-architecture)
7. [Model Architecture](#model-architecture)
8. [API Design](#api-design)
9. [Security & Privacy](#security--privacy)
10. [Scalability Considerations](#scalability-considerations)

---

## System Overview

SafeMind AI implements a **3-tier client-server architecture** with specialized middleware for mental health safety and cultural adaptation.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Client Layer                        │
│  (Vue.js SPA - Responsive Web Application)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTPS/REST
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (FastAPI - RESTful API Server)                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐       │
│  │   Safety     │  │   Cultural   │  │  Context    │       │
│  │  Detection   │  │  Adaptation  │  │  Manager    │       │
│  └──────────────┘  └──────────────┘  └─────────────┘       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                        AI Layer                             │
│  (Fine-tuned LLM with LoRA Adapters)                        │
│                                                              │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  Base Model    │ + LoRA →│  Mental Health   │           │
│  │ (Phi-3/GPT)    │         │     Adapter      │           │
│  └────────────────┘         └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture Diagram

### Detailed System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Vue.js)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐        │
│  │  ChatWindow  │  │  Resources  │  │  UI Components │        │
│  │  Component   │  │    Modal    │  │   (Shared)     │        │
│  └──────┬───────┘  └──────┬──────┘  └────────┬───────┘        │
│         │                 │                   │                │
│         └─────────────────┴───────────────────┘                │
│                           │                                    │
│                    ┌──────▼──────┐                             │
│                    │  API Service │                            │
│                    │  (Axios)     │                            │
│                    └──────┬───────┘                            │
└───────────────────────────┼────────────────────────────────────┘
                            │ HTTP POST /api/chat
                            │ JSON: {message, session_id, culture}
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Request Handler (FastAPI Route)              │ │
│  │  POST /api/chat → chat(ChatRequest) → ChatResponse       │ │
│  └────────┬──────────────────────────────────────────────────┘ │
│           │                                                     │
│           ↓                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Step 1: Context Manager                                │    │
│  │ • Retrieve or create session                           │    │
│  │ • Load conversation history                            │    │
│  │ • Build context summary                                │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                     │
│           ↓                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Step 2: Safety Detection (9 Layers)                    │    │
│  │ • Layer 1: Immediate risk keywords                     │    │
│  │ • Layer 2: High risk keywords                          │    │
│  │ • Layer 3: Medium risk keywords                        │    │
│  │ • Layer 4: Pattern matching (regex)                    │    │
│  │ • Layer 5: Sentiment analysis                          │    │
│  │ • Layer 6: Contextual indicators                       │    │
│  │ • Layer 7: Temporal urgency                            │    │
│  │ • Layer 8: Planning indicators                         │    │
│  │ • Layer 9: Means access detection                      │    │
│  │ → Output: {risk_level, confidence, triggers}           │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                     │
│           ↓                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Step 3: AI Model (Conditional)                         │    │
│  │                                                         │    │
│  │ ┌──────────────────────────────────────────┐           │    │
│  │ │ Base Model (Phi-3 / GPT / DialoGPT)      │           │    │
│  │ │   +                                       │           │    │
│  │ │ LoRA Adapters (Fine-tuned)                │           │    │
│  │ │                                           │           │    │
│  │ │ Input: Prompt Engineering                 │           │    │
│  │ │ • Instruction (empathy, no diagnosis)     │           │    │
│  │ │ • User message                            │           │    │
│  │ │ • Conversation context                    │           │    │
│  │ │ • Risk level                              │           │    │
│  │ │                                           │           │    │
│  │ │ Output: Empathetic response               │           │    │
│  │ └──────────────────────────────────────────┘           │    │
│  │                                                         │    │
│  │ Fallback: Template-based responses                     │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                     │
│           ↓                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Step 4: Safety Response Injection                      │    │
│  │ • If risk_level >= high:                               │    │
│  │   - Prepend crisis intervention message                │    │
│  │   - Include emergency hotlines (1333, 119)             │    │
│  │   - Add safety resources                               │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                     │
│           ↓                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Step 5: Cultural Adaptation                            │    │
│  │ • Apply Sri Lankan context:                            │    │
│  │   - Family dynamics                                    │    │
│  │   - Academic pressure                                  │    │
│  │   - Social stigma awareness                            │    │
│  │   - Cultural resources                                 │    │
│  │ • Adjust tone for cultural appropriateness             │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                     │
│           ↓                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Step 6: Context Update                                 │    │
│  │ • Save user message                                    │    │
│  │ • Save bot response                                    │    │
│  │ • Update emotion tracking                              │    │
│  │ • Update risk trend                                    │    │
│  └────────┬───────────────────────────────────────────────┘    │
│           │                                                     │
│           ↓                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ Response                                                │    │
│  │ JSON: {response, session_id, safety, timestamp, ...}   │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP 200 OK (JSON)
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND (Display)                          │
│  • Render bot message                                           │
│  • Show risk badge if applicable                                │
│  • Display crisis alert if high risk                            │
│  • Auto-scroll to latest message                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### Frontend Components

#### 1. Vue.js Application (`frontend-vue/`)

**Main Components:**

- **App.vue**
  - Root component
  - Manages global state
  - Handles connection status
  - Shows disclaimer banner

- **ChatWindow.vue**
  - Message display (user/bot)
  - Input area with keyboard shortcuts
  - Typing indicator
  - Risk level badges
  - Auto-scroll functionality

- **ResourcesModal.vue**
  - Emergency resources display
  - Crisis hotlines (1333, 119, etc.)
  - Mental health resources
  - Triggered on high-risk detection

**Services:**

- **api.js**
  - Axios HTTP client
  - API endpoint wrappers
  - Request/response interceptors
  - Error handling

**Utilities:**

- **helpers.js**
  - Session ID generation
  - Time formatting
  - Risk level utilities
  - Local storage management

### Backend Components

#### 2. FastAPI Server (`backend/app_fastapi.py`)

**Core Modules:**

**Safety Detection** (`safety_detector.py`, `enhanced_safety_detector.py`)
```python
class EnhancedSafetyDetector:
    • detect_crisis(text, context) → {risk_level, confidence, triggers}
    • 9 detection layers
    • Multi-weighted scoring
    • Contextual analysis
```

**AI Model** (`ai_model.py`, `ai_model_free.py`)
```python
class SafeMindAI:
    • generate_response(message, context, risk_level) → response
    • Supports: OpenAI, Hugging Face, Local models
    • Fallback to templates
    • Prompt engineering for empathy
```

**Context Manager** (`context_manager.py`)
```python
class ContextManager:
    • get_or_create_session(session_id)
    • add_message(user_msg, bot_msg, emotion, risk)
    • get_context_summary() → {history, emotion, risk_trend}
    • In-memory session storage
```

**Cultural Adapter** (`cultural_adapter.py`)
```python
class CulturalAdapter:
    • adapt_response(response, culture)
    • Sri Lankan context integration
    • Cultural resource mapping
    • Tone adjustment
```

---

## Data Flow

### Message Processing Flow

```
1. User Input
   ↓
   Frontend validates input
   ↓
2. HTTP Request
   POST /api/chat
   {
     "message": "I feel anxious",
     "session_id": "session-123",
     "culture": "south_asian"
   }
   ↓
3. Backend Receives Request
   ↓
4. Get Session Context
   • Load conversation history
   • Build context summary
   ↓
5. Safety Detection (9 layers)
   • Keyword matching
   • Pattern analysis
   • Sentiment scoring
   • Risk calculation
   ↓
   {
     "risk_level": "medium",
     "confidence": 0.78,
     "triggers": ["anxious"],
     "requires_intervention": false
   }
   ↓
6. AI Generation
   • Build prompt with context
   • Call AI model (local or API)
   • Generate empathetic response
   ↓
   "I hear that you're feeling anxious..."
   ↓
7. Safety Response (if needed)
   If risk_level >= high:
   • Prepend crisis message
   • Add emergency resources
   ↓
8. Cultural Adaptation
   • Apply Sri Lankan context
   • Adjust tone
   • Add cultural resources
   ↓
9. Context Update
   • Save user message
   • Save bot response
   • Update emotion tracking
   • Update risk trend
   ↓
10. Return Response
    {
      "response": "...",
      "session_id": "session-123",
      "safety": {...},
      "timestamp": "2026-01-05T10:30:00Z",
      "ai_powered": true
    }
    ↓
11. Frontend Displays
    • Render message
    • Show risk badge
    • Trigger modal if crisis
```

---

## Technology Stack

### Frontend Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Vue.js** | UI framework | 3.3.8 |
| **Vite** | Build tool | 5.0.4 |
| **Axios** | HTTP client | 1.6.2 |
| **Pinia** | State management | 2.1.7 |

### Backend Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming language | 3.9+ |
| **FastAPI** | Web framework | 0.104.1 |
| **Uvicorn** | ASGI server | 0.24.0 |
| **Pydantic** | Data validation | 2.5.0 |

### AI/ML Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| **Transformers** | Model loading | 4.35.2 |
| **PyTorch** | Deep learning | 2.1.1 |
| **PEFT** | LoRA fine-tuning | 0.7.0 |
| **Accelerate** | Training optimization | 0.25.0 |
| **TextBlob** | Sentiment analysis | 0.17.1 |

### Model Options

| Model | Size | Use Case | Cost |
|-------|------|----------|------|
| **Phi-3 Mini** | 3.8B | Best quality local | Free |
| **DialoGPT** | 345M | Fast, conversational | Free |
| **OpenAI GPT-3.5** | N/A | Highest quality | ~$0.002/msg |
| **Claude Sonnet** | N/A | High quality | ~$0.003/msg |

---

## Safety Architecture

### 9-Layer Detection System

```
Layer 1: Immediate Keywords
├─ "kill myself", "suicide", "end my life"
└─ Weight: 1.0 (highest priority)

Layer 2: High Risk Keywords
├─ "want to die", "hopeless", "no point living"
└─ Weight: 0.8

Layer 3: Medium Risk Keywords
├─ "worthless", "burden", "hate myself"
└─ Weight: 0.5

Layer 4: Pattern Matching (Regex)
├─ "(no|any) .{0,10} (point|reason) .{0,10} (living|life)"
└─ Weight: 0.7

Layer 5: Sentiment Analysis
├─ TextBlob polarity score
└─ Weight: 0.3 (if highly negative)

Layer 6: Contextual Indicators
├─ Hopelessness, isolation, burden themes
└─ Weight: 0.6

Layer 7: Temporal Urgency
├─ "tonight", "now", "today"
└─ Weight: 0.9

Layer 8: Planning Indicators
├─ "plan to", "thinking of", "going to"
└─ Weight: 0.95

Layer 9: Means Access
├─ References to pills, weapons, etc.
└─ Weight: 1.0
```

**Risk Calculation:**
```python
total_score = sum(layer_weight * layer_detection)
risk_level = categorize_risk(total_score)

# Risk levels:
# none: 0.0
# low: 0.1 - 0.3
# medium: 0.3 - 0.6
# high: 0.6 - 0.8
# immediate: > 0.8
```

---

## Model Architecture

### Fine-Tuning with LoRA

```
Base Model (Frozen)
├─ Embedding Layer
├─ Transformer Blocks (12-32 layers)
│  ├─ Self-Attention
│  │  ├─ Q, K, V projections ← LoRA adapters injected here
│  │  └─ Output projection ← LoRA adapters injected here
│  └─ Feed-Forward Network
└─ Output Layer

LoRA Adapters (Trainable)
├─ Low-rank matrices: A (d × r), B (r × d)
├─ Rank r = 8 (typically)
├─ Only ~0.5% of model parameters
└─ Fast training, small file size

Inference
├─ Load base model
├─ Load LoRA adapters
├─ Merge: W' = W + α * (B * A)
└─ Generate response
```

---

## API Design

### RESTful Endpoints

| Endpoint | Method | Purpose | Request | Response |
|----------|--------|---------|---------|----------|
| `/api/chat` | POST | Send message | `{message, session_id, culture}` | `{response, safety, ...}` |
| `/api/health` | GET | Health check | - | `{status, system}` |
| `/api/resources` | GET | Get resources | `?culture=south_asian` | `{emergency, support, ...}` |
| `/api/session/{id}` | GET | Get session | - | `{context, history}` |
| `/api/session/{id}/export` | GET | Export session | - | `{conversation, mood, risk}` |
| `/api/test` | POST | Test endpoint | `{message}` | `{input, process, output}` |

### Request/Response Schema

**Chat Request:**
```json
{
  "message": "I feel anxious",
  "session_id": "session-123",
  "culture": "south_asian"
}
```

**Chat Response:**
```json
{
  "response": "I hear that you're feeling anxious...",
  "session_id": "session-123",
  "safety": {
    "risk_level": "low",
    "requires_intervention": false,
    "confidence": 0.65
  },
  "timestamp": "2026-01-05T10:30:00Z",
  "ai_powered": true,
  "message_count": 5
}
```

---

## Security & Privacy

### Security Measures

1. **Input Validation**
   - Pydantic models for request validation
   - Max message length enforcement
   - SQL injection prevention (no database yet)

2. **CORS Configuration**
   - Specific origin whitelisting (production)
   - Credential support
   - Method restrictions

3. **Rate Limiting** (planned)
   - Per-IP request limits
   - Session-based throttling
   - API key quotas

4. **Data Privacy**
   - No persistent storage (currently)
   - In-memory sessions only
   - No PII collection
   - HTTPS enforcement (production)

### Privacy Considerations

- **Session Data**: Stored in-memory, lost on restart
- **Logs**: Should not contain user messages (configure in production)
- **Third-party APIs**: User data sent to OpenAI/Anthropic if using their APIs
- **Local Model**: Fully private, data never leaves server

---

## Scalability Considerations

### Current Limitations

- In-memory session storage (not persistent)
- Single-server deployment
- No load balancing
- No caching layer

### Scaling Path

**Phase 1: Vertical Scaling**
- Increase server resources
- Optimize model inference
- Add request caching

**Phase 2: Database Integration**
```
Application Server
    ↓
PostgreSQL / MongoDB
├─ User sessions
├─ Conversation history
└─ Analytics data
```

**Phase 3: Horizontal Scaling**
```
Load Balancer (nginx)
    ↓
┌───────┬───────┬───────┐
│ App 1 │ App 2 │ App 3 │
└───┬───┴───┬───┴───┬───┘
    └───────┴───────┘
          ↓
   Shared Database
   (PostgreSQL)
          ↓
   Model Server(s)
   (GPU instances)
```

**Phase 4: Microservices**
```
API Gateway
    ↓
┌────────────┬──────────────┬──────────────┐
│   Chat     │   Safety     │   Model      │
│  Service   │  Detection   │  Inference   │
└────────────┴──────────────┴──────────────┘
    ↓              ↓               ↓
Database       Redis Cache    GPU Cluster
```

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Response Time | <3s | ~2.3s |
| Concurrent Users | 100+ | 10-20 |
| Uptime | 99.9% | N/A |
| Model Inference | <1s | ~0.5s |
| Safety Detection | <100ms | ~50ms |

---

**Document Version:** 2.0
**Last Updated:** January 5, 2026
**Maintained by:** SafeMind AI Team

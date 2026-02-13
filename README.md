# SafeMind - Mental Health Support Platform

A conversational mental health support system designed specifically for the Sri Lankan socio-cultural context. Built as part of a BSc Software Engineering thesis at Staffordshire University.

## Overview

SafeMind provides empathetic, context-aware mental health support through an interactive chat interface. The system understands cultural nuances specific to Sri Lanka, including academic pressures (A/L exams), family dynamics, and societal stigma around mental health.

## Key Features

### Crisis Detection System
- Multi-layered risk assessment using 11 detection layers
- Keyword-based pattern matching for risk indicators
- Sentiment analysis for emotional state assessment
- Contextual understanding of user messages
- Cultural pressure recognition (family expectations, academic stress)
- Temporal urgency detection ("tonight", "now")
- Planning indicators and means access detection
- Real-time risk level classification (low/medium/high/immediate)

### Cultural Adaptation
- Sri Lankan context awareness (A/L exam stress, family pressure)
- Integration with local emergency services (1333 hotline, Sumithrayo)
- Understanding of South Asian family dynamics
- Recognition of cultural stigma around mental health
- Community and spiritual support emphasis

### User Interface
- Real-time chat interface with conversation history
- Mood tracking and visualization
- Emergency resource panel with local hotline numbers
- Crisis alert system with immediate intervention protocols
- Retro pixel-art themed design

### Session Management
- Conversation context tracking across messages
- Session persistence for continuity
- Message history storage
- User mood trend analysis

## Technology Stack

### Backend
- Python 3.11+ with Flask web framework
- Natural language processing and sentiment analysis
- Machine learning for response generation
- Multi-backend architecture supporting various conversation engines
- Crisis detection running independently of external services

### Frontend
- React 18.2.0 for the user interface
- Axios for API communication
- Recharts for mood visualization
- Lucide React for iconography

### Data Management
- Training dataset with 1,500+ mental health conversations
- Crisis pattern database with keywords and contextual indicators
- Response template library organized by topic
- Cultural context configuration for Sri Lankan adaptation

## Project Structure

```
MIDPOINT/
├── backend/
│   ├── app_improved.py            # Main Flask application server
│   ├── ai_model_free.py           # Response generation system
│   ├── enhanced_safety_detector.py # Multi-layer crisis detection
│   ├── context_manager.py         # Session and context management
│   ├── cultural_adapter.py        # Cultural adaptation logic
│   ├── config.py                  # Configuration management
│   ├── train_model.py             # Model training utilities
│   ├── test_mvp.py                # Automated testing suite
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js                 # Main application component
│   │   ├── App.css                # Application styling
│   │   ├── components/
│   │   │   ├── ChatInterface.js   # Chat UI component
│   │   │   ├── MessageBubble.js   # Message rendering
│   │   │   ├── SafetyAlert.js     # Crisis alert modal
│   │   │   ├── MoodTracker.js     # Mood visualization
│   │   │   └── ResourcePanel.js   # Emergency resource display
│   │   └── services/
│   │       └── api.js             # Backend API integration
│   └── package.json
├── data/
│   ├── mental_health_dataset.json     # Training data
│   ├── enhanced_crisis_patterns.json  # Crisis detection patterns
│   ├── response_templates.json        # Response templates
│   └── cultural_templates.json        # Cultural context data
└── scripts/
    └── expand_dataset.py              # Dataset generation tool
```

## Quick Start

### Backend Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Configure .env with your settings
python3 app_improved.py
```

The backend server will start on `http://localhost:5001`

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

The application will open at `http://localhost:3000`

## Configuration

The system supports multiple conversation backends configurable via environment variables:

- External service integration (requires API key)
- Trained local model (requires model training)
- Template-based responses (no dependencies)

Configuration options are set in `backend/.env` following the template in `.env.example`

## Testing

Run the automated test suite:

```bash
cd backend
source venv/bin/activate
python3 test_mvp.py
```

Tests cover:
- Crisis detection accuracy across risk levels
- Cultural context understanding
- Academic stress handling
- Response appropriateness
- Emergency escalation protocols

## Safety & Compliance

**Important**: SafeMind is not a replacement for professional mental health care.

### Emergency Resources (Sri Lanka)
- **Crisis Hotline**: 1333
- **Sumithrayo**: 011-2696666
- **Emergency Services**: 119

The system includes built-in safeguards:
- Real-time crisis detection
- Automatic emergency resource display
- No diagnosis or medication prescription
- Clear disclaimers about professional care
- Immediate escalation for high-risk situations

## Dataset

The training dataset includes:
- 1,500+ mental health conversation examples
- Culturally relevant scenarios (A/L stress, family conflicts)
- Crisis situations with appropriate responses
- Positive and recovery-focused conversations
- Academic and career stress contexts

## Model Training

For information on training and deploying custom models, see `set-up-project.md`

## License

Academic project for Staffordshire University BSc Software Engineering.

## Support

For system setup and configuration, refer to `set-up-project.md`

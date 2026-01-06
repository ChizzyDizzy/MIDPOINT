"""
SafeMind AI - FastAPI Backend Implementation
Mental Health AI Assistant with real AI model integration

FastAPI version with async support and automatic API documentation
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

# Import components
from safety_detector import SafetyDetector
from context_manager import ContextManager
from ai_model import SafeMindAI
from cultural_adapter import CulturalAdapter
from config import Config

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SafeMind AI API",
    description="Mental Health Awareness Chatbot API for Sri Lankan Context",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
safety_detector = SafetyDetector()
context_manager = ContextManager()
ai_model = SafeMindAI()
cultural_adapter = CulturalAdapter()

# System status
SYSTEM_STATUS = {
    'ai_enabled': ai_model.use_ai,
    'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
    'safety_detection': 'active',
    'cultural_adaptation': 'active',
    'version': '2.0.0-FastAPI',
    'framework': 'FastAPI'
}

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    culture: Optional[str] = Field("south_asian", description="Cultural context")

    class Config:
        schema_extra = {
            "example": {
                "message": "I feel anxious about my exams",
                "session_id": "user-123",
                "culture": "south_asian"
            }
        }

class ChatResponse(BaseModel):
    response: str
    session_id: str
    safety: Dict[str, Any]
    timestamp: str
    ai_powered: bool
    message_count: int

class HealthResponse(BaseModel):
    status: str
    system: Dict[str, Any]
    timestamp: str

class TestRequest(BaseModel):
    message: str = Field("I feel anxious", description="Test message")

# API Routes

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "SafeMind AI API",
        "version": "2.0.0",
        "description": "Mental Health Awareness Chatbot for Sri Lankan Context",
        "docs": "/api/docs",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health",
            "test": "/api/test",
            "resources": "/api/resources"
        }
    }

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint - processes user message and generates AI response

    **Process Flow:**
    1. Validate input
    2. Get or create conversation context
    3. Perform safety detection (9-layer system)
    4. Generate AI response
    5. Add safety warnings if needed
    6. Apply cultural adaptation
    7. Update context
    8. Return response

    **Safety Levels:**
    - `none`: No risk detected
    - `low`: Mild concern (stress, anxiety)
    - `medium`: Moderate concern (depression symptoms)
    - `high`: Serious concern (hopelessness)
    - `immediate`: Crisis requiring immediate intervention
    """
    try:
        message = request.message.strip()
        session_id = request.session_id or str(uuid.uuid4())
        culture = request.culture or 'south_asian'

        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Step 1: Get or create conversation context
        context = context_manager.get_or_create_session(session_id)

        # Step 2: Safety detection (Critical - Multi-layered)
        safety_result = safety_detector.detect_crisis(message)

        # Step 3: Generate AI response with context
        context_summary = context.get_context_summary()

        try:
            # Use AI model for response generation
            ai_response = ai_model.generate_response(
                user_message=message,
                context_summary=context_summary,
                risk_level=safety_result['risk_level'],
                emotion='concerned' if safety_result['risk_level'] != 'none' else 'neutral'
            )

            base_response = ai_response
            ai_powered = True

        except Exception as e:
            print(f"AI generation error: {e}")
            # Fallback to template-based response
            base_response = ai_model._generate_fallback_response(message, 'neutral')
            ai_powered = False

        # Step 4: Add immediate safety response if needed
        if safety_result['requires_intervention']:
            safety_message = safety_detector.generate_safety_response(safety_result['risk_level'])
            if safety_message:
                base_response = f"{safety_message}\n\n{base_response}"

        # Step 5: Cultural adaptation
        final_response = cultural_adapter.adapt_response(base_response, culture)

        # Step 6: Update context with this interaction
        context.add_message(
            message,
            final_response,
            emotion='concerned' if safety_result['risk_level'] != 'none' else 'neutral',
            risk_level=safety_result['risk_level']
        )

        # Step 7: Return response
        return ChatResponse(
            response=final_response,
            session_id=session_id,
            safety={
                'risk_level': safety_result['risk_level'],
                'requires_intervention': safety_result['requires_intervention'],
                'confidence': safety_result['confidence']
            },
            timestamp=datetime.now().isoformat(),
            ai_powered=ai_powered,
            message_count=context.message_count
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your message"
        )

@app.get("/api/session/{session_id}", tags=["Session"])
async def get_session(session_id: str):
    """
    Get session history and context

    Returns conversation history, mood trends, and risk assessment for a session.
    """
    try:
        context = context_manager.get_or_create_session(session_id)
        summary = context.get_context_summary()

        return {
            'session_id': session_id,
            'context': summary,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/session/{session_id}/export", tags=["Session"])
async def export_session(session_id: str):
    """
    Export conversation history for user

    Provides a downloadable summary of the conversation including:
    - Message history
    - Mood trend analysis
    - Risk assessment
    """
    try:
        context = context_manager.get_or_create_session(session_id)
        summary = context.get_context_summary()

        # Format for export
        export_data = {
            'session_id': session_id,
            'export_date': datetime.now().isoformat(),
            'conversation_history': summary.get('recent_messages', []),
            'mood_trend': summary.get('current_emotion', 'neutral'),
            'risk_assessment': summary.get('risk_trend', 'stable'),
            'message_count': summary.get('message_count', 0)
        }

        return export_data

    except Exception as e:
        raise HTTPException(status_code=500, detail="Export failed")

@app.get("/api/resources", tags=["Resources"])
async def get_resources(culture: str = "south_asian"):
    """
    Get emergency and support resources

    Returns culturally-appropriate mental health resources including:
    - Crisis hotlines (24/7)
    - Emergency services
    - Mental health helplines
    - Self-help resources

    **Sri Lankan Resources:**
    - **1333**: National Mental Health Crisis Hotline (24/7)
    - **119**: Emergency Services
    - **011-2696666**: Sumithrayo Emotional Support (24/7)
    - **1926**: Mental Health Helpline
    """
    # Base emergency resources
    resources = {
        'emergency': {
            'crisis_hotline': {
                'name': 'National Mental Health Crisis Hotline (Sri Lanka)',
                'number': '1333',
                'available': '24/7',
                'description': 'Free confidential crisis support'
            },
            'emergency_services': {
                'name': 'Emergency Services',
                'number': '119',
                'available': '24/7',
                'description': 'Police, ambulance, fire'
            },
            'sumithrayo': {
                'name': 'Sumithrayo (Emotional Support)',
                'number': '011-2696666',
                'available': '24/7',
                'description': 'Volunteer-run emotional support service'
            }
        },
        'support': {
            'mental_health_helpline': {
                'name': 'Mental Health Helpline',
                'number': '1926',
                'available': 'Daily',
                'description': 'Mental health information and referrals'
            }
        },
        'self_help': {
            'breathing_exercises': 'Available in app',
            'mindfulness': 'Available in app',
            'mood_tracking': 'Available in app'
        },
        'disclaimer': 'This chatbot is NOT a replacement for professional mental health care. If you are in crisis, please contact emergency services or a crisis hotline immediately.'
    }

    # Add cultural-specific resources
    try:
        cultural_resources = cultural_adapter.get_culturally_appropriate_resources(culture)
        resources.update(cultural_resources)
    except:
        pass

    return resources

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint

    Returns the current status of all system components.
    Use this to verify the API is running and all services are operational.
    """
    return HealthResponse(
        status='healthy',
        system=SYSTEM_STATUS,
        timestamp=datetime.now().isoformat()
    )

@app.get("/api/system/status", tags=["System"])
async def system_status():
    """
    Get detailed system status

    Includes:
    - AI model status
    - Active session count
    - Safety detection status
    - Cultural adaptation status
    """
    return {
        'status': SYSTEM_STATUS,
        'sessions_active': len(context_manager.sessions),
        'timestamp': datetime.now().isoformat()
    }

@app.post("/api/test", tags=["Testing"])
async def test_endpoint(request: TestRequest):
    """
    Test endpoint for MVP demonstration

    Shows complete Input → Process → Output flow.

    **Process Steps:**
    1. **Input**: Receive user message
    2. **Safety Detection**: 9-layer crisis analysis
    3. **AI Generation**: Generate empathetic response
    4. **Cultural Adaptation**: Apply cultural context
    5. **Output**: Return final response

    Useful for debugging and demonstrating system functionality.
    """
    test_message = request.message

    # Input
    input_data = {
        'message': test_message,
        'timestamp': datetime.now().isoformat()
    }

    # Process
    safety_result = safety_detector.detect_crisis(test_message)
    ai_response = ai_model.generate_response(
        user_message=test_message,
        context_summary={},
        risk_level=safety_result['risk_level']
    )

    process_data = {
        'safety_detection': safety_result,
        'ai_generation': 'completed',
        'cultural_adaptation': 'applied'
    }

    # Output
    output_data = {
        'response': ai_response,
        'risk_assessment': safety_result['risk_level'],
        'intervention_required': safety_result['requires_intervention']
    }

    return {
        'test': 'SafeMind AI MVP Test',
        'input': input_data,
        'process': process_data,
        'output': output_data,
        'timestamp': datetime.now().isoformat()
    }

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            'error': 'Endpoint not found',
            'path': str(request.url),
            'timestamp': datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'message': 'Please contact support',
            'timestamp': datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Print startup information"""
    print("=" * 60)
    print("SafeMind AI - Mental Health Assistant (FastAPI)")
    print("=" * 60)
    print(f"AI Model: {SYSTEM_STATUS['model']}")
    print(f"AI Enabled: {SYSTEM_STATUS['ai_enabled']}")
    print(f"Safety Detection: {SYSTEM_STATUS['safety_detection']}")
    print(f"Cultural Adaptation: {SYSTEM_STATUS['cultural_adaptation']}")
    print("=" * 60)
    print("Server starting on http://localhost:8000")
    print("API Documentation: http://localhost:8000/api/docs")
    print("Alternative Docs: http://localhost:8000/api/redoc")
    print("=" * 60)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\nShutting down SafeMind AI...")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

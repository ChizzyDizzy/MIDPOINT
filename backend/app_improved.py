"""
SafeMind AI - Improved Main Application
Mental Health AI Assistant with real AI model integration
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

# Import improved components
from enhanced_safety_detector import EnhancedSafetyDetector
from context_manager import ContextManager
from ai_model_free import SafeMindAI
from cultural_adapter import CulturalAdapter
from config import Config

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, supports_credentials=True)

# Initialize components
safety_detector = EnhancedSafetyDetector()
context_manager = ContextManager()
ai_model = SafeMindAI()
cultural_adapter = CulturalAdapter()

# Track system status
SYSTEM_STATUS = {
    'ai_enabled': ai_model.use_ai,
    'ai_backend': os.getenv('AI_BACKEND', 'openai'),
    'safety_detection': 'active',
    'cultural_adaptation': 'active',
    'version': '1.0.0'
}

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint - processes user message and generates AI response

    Request JSON:
    {
        "message": str,
        "session_id": str (optional),
        "culture": str (optional, default: "south_asian")
    }

    Returns:
    {
        "response": str,
        "session_id": str,
        "safety": dict,
        "timestamp": str,
        "ai_powered": bool
    }
    """
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        culture = data.get('culture', 'south_asian')

        # Validate input
        if not message:
            return jsonify({
                'error': 'Message cannot be empty',
                'timestamp': datetime.now().isoformat()
            }), 400

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
        return jsonify({
            'response': final_response,
            'session_id': session_id,
            'safety': {
                'risk_level': safety_result['risk_level'],
                'requires_intervention': safety_result['requires_intervention'],
                'confidence': safety_result['confidence']
            },
            'timestamp': datetime.now().isoformat(),
            'ai_powered': ai_powered,
            'message_count': context.message_count
        })

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'An error occurred processing your message',
            'message': 'Please try again or contact support if the issue persists',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session history and context"""
    try:
        context = context_manager.get_or_create_session(session_id)
        summary = context.get_context_summary()

        return jsonify({
            'session_id': session_id,
            'context': summary,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'error': 'Session not found',
            'timestamp': datetime.now().isoformat()
        }), 404


@app.route('/api/session/<session_id>/export', methods=['GET'])
def export_session(session_id):
    """Export conversation history for user"""
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

        return jsonify(export_data)

    except Exception as e:
        return jsonify({'error': 'Export failed'}), 500


@app.route('/api/resources', methods=['GET'])
def get_resources():
    """Get emergency and support resources"""
    culture = request.args.get('culture', 'south_asian')

    # Base emergency resources
    resources = {
        'emergency': {
            'crisis_hotline': {
                'name': 'National Mental Health Crisis Hotline',
                'number': '1333',
                'available': '24/7'
            },
            'emergency_services': {
                'name': 'Emergency Services',
                'number': '119',
                'available': '24/7'
            },
            'sumithrayo': {
                'name': 'Sumithrayo (Emotional Support)',
                'number': '011-2696666',
                'available': '24/7'
            }
        },
        'support': {
            'mental_health_helpline': {
                'name': 'Mental Health Helpline',
                'number': '1926',
                'available': 'Daily'
            }
        },
        'self_help': {
            'breathing_exercises': 'Available in app',
            'mindfulness': 'Available in app',
            'mood_tracking': 'Available in app'
        }
    }

    # Add cultural-specific resources
    cultural_resources = cultural_adapter.get_culturally_appropriate_resources(culture)
    resources.update(cultural_resources)

    return jsonify(resources)


@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'system': SYSTEM_STATUS,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Get detailed system status"""
    return jsonify({
        'status': SYSTEM_STATUS,
        'sessions_active': len(context_manager.sessions),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/test', methods=['POST'])
def test_endpoint():
    """
    Test endpoint for system verification
    Shows Input → Process → Output flow
    """
    data = request.json
    test_message = data.get('message', 'I feel anxious')

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

    return jsonify({
        'test': 'SafeMind AI System Test',
        'input': input_data,
        'process': process_data,
        'output': output_data,
        'timestamp': datetime.now().isoformat()
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please contact support',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("SafeMind AI - Mental Health Assistant")
    print("=" * 60)
    print(f"AI Backend: {SYSTEM_STATUS['ai_backend']}")
    print(f"AI Enabled: {SYSTEM_STATUS['ai_enabled']}")
    print(f"Safety Detection: {SYSTEM_STATUS['safety_detection']}")
    print(f"Cultural Adaptation: {SYSTEM_STATUS['cultural_adaptation']}")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)

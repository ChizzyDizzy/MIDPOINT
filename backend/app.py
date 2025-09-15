from flask import Flask, request, jsonify, session
from flask_cors import CORS
import uuid
from datetime import datetime

from safety_detector import SafetyDetector
from context_manager import ContextManager
from response_generator import ResponseGenerator
from cultural_adapter import CulturalAdapter
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, supports_credentials=True)

# Initialize components
safety_detector = SafetyDetector()
context_manager = ContextManager()
response_generator = ResponseGenerator()
cultural_adapter = CulturalAdapter()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    
    # Get or create context
    context = context_manager.get_or_create_session(session_id)
    
    # Safety check
    safety_result = safety_detector.detect_crisis(message)
    
    # Generate base response
    context_summary = context.get_context_summary()
    base_response = response_generator.generate_response(
        message, 
        context_summary,
        emotion='concerned' if safety_result['risk_level'] != 'none' else 'neutral'
    )
    
    # Add safety response if needed
    safety_response = safety_detector.generate_safety_response(safety_result['risk_level'])
    if safety_response:
        base_response = f"{safety_response}\n\n{base_response}"
    
    # Cultural adaptation
    final_response = cultural_adapter.adapt_response(base_response)
    
    # Update context
    context.add_message(
        message, 
        final_response, 
        emotion='concerned' if safety_result['risk_level'] != 'none' else 'neutral',
        risk_level=safety_result['risk_level']
    )
    
    return jsonify({
        'response': final_response,
        'session_id': session_id,
        'safety': safety_result,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session history"""
    context = context_manager.get_or_create_session(session_id)
    return jsonify(context.get_context_summary())

@app.route('/api/resources', methods=['GET'])
def get_resources():
    """Get emergency resources"""
    culture = request.args.get('culture', 'south_asian')
    resources = cultural_adapter.get_culturally_appropriate_resources(culture)
    resources.update(Config.EMERGENCY_RESOURCES)
    return jsonify(resources)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
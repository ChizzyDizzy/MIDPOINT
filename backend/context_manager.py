from datetime import datetime
from typing import List, Dict
import json

class ConversationContext:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history = []
        self.emotional_states = []
        self.topics = []
        self.start_time = datetime.now()
        self.risk_history = []
        
    def add_message(self, message: str, response: str, emotion: str = 'neutral', risk_level: str = 'none'):
        """Add a message exchange to history"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'bot_response': response,
            'emotion': emotion,
            'risk_level': risk_level
        }
        self.conversation_history.append(entry)
        self.emotional_states.append(emotion)
        self.risk_history.append(risk_level)
        
    def get_context_summary(self) -> Dict:
        """Get summary of conversation context"""
        return {
            'session_id': self.session_id,
            'message_count': len(self.conversation_history),
            'duration': (datetime.now() - self.start_time).seconds,
            'current_emotion': self.emotional_states[-1] if self.emotional_states else 'neutral',
            'risk_trend': self._calculate_risk_trend(),
            'recent_messages': self.conversation_history[-3:] if len(self.conversation_history) >= 3 else self.conversation_history
        }
    
    def _calculate_risk_trend(self) -> str:
        """Calculate if risk is escalating, stable, or decreasing"""
        if len(self.risk_history) < 2:
            return 'stable'
        
        risk_values = {'none': 0, 'low': 1, 'medium': 2, 'high': 3, 'immediate': 4}
        recent_risks = [risk_values.get(r, 0) for r in self.risk_history[-5:]]
        
        if len(recent_risks) >= 2:
            if recent_risks[-1] > recent_risks[-2]:
                return 'escalating'
            elif recent_risks[-1] < recent_risks[-2]:
                return 'decreasing'
        
        return 'stable'
    
    def get_conversation_for_ai(self) -> str:
        """Format conversation history for AI context"""
        if not self.conversation_history:
            return "This is the beginning of the conversation."
        
        context = "Previous conversation:\n"
        for entry in self.conversation_history[-5:]:  # Last 5 exchanges
            context += f"User: {entry['user_message']}\n"
            context += f"Assistant: {entry['bot_response']}\n"
        
        return context

class ContextManager:
    def __init__(self):
        self.sessions = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(session_id)
        return self.sessions[session_id]
    
    def clear_old_sessions(self):
        """Clear sessions older than 24 hours"""
        current_time = datetime.now()
        to_remove = []
        
        for session_id, context in self.sessions.items():
            if (current_time - context.start_time).days >= 1:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
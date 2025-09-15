import json
import random
from typing import Dict, List
from textblob import TextBlob

class ResponseGenerator:
    def __init__(self):
        with open('../data/response_templates.json', 'r') as f:
            self.templates = json.load(f)
        
        # For future: This is where ChatGPT API would be integrated
        self.use_ai = False  # Set to True when API is available
    
    def generate_response(self, message: str, context: Dict, emotion: str = 'neutral') -> str:
        """Generate appropriate response based on message and context"""
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(message)
        
        # Determine response category
        category = self._categorize_message(message, sentiment)
        
        # Select appropriate template
        if category in self.templates:
            response_options = self.templates[category]
            base_response = random.choice(response_options)
        else:
            base_response = random.choice(self.templates['general'])
        
        # Personalize based on context
        response = self._personalize_response(base_response, context)
        
        # Add follow-up question to encourage dialogue
        if category != 'crisis':
            response += " " + self._get_follow_up_question(category)
        
        return response
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def _categorize_message(self, message: str, sentiment: float) -> str:
        """Categorize message for appropriate response"""
        message_lower = message.lower()
        
        # Check for specific conditions
        if any(word in message_lower for word in ['anxious', 'anxiety', 'worried', 'nervous']):
            return 'anxiety'
        elif any(word in message_lower for word in ['sad', 'depressed', 'hopeless', 'down']):
            return 'depression'
        elif any(word in message_lower for word in ['angry', 'frustrated', 'mad', 'upset']):
            return 'anger'
        elif any(word in message_lower for word in ['lonely', 'alone', 'isolated']):
            return 'loneliness'
        elif any(word in message_lower for word in ['stressed', 'overwhelmed', 'pressure']):
            return 'stress'
        elif sentiment > 0.5:
            return 'positive'
        elif sentiment < -0.5:
            return 'negative'
        else:
            return 'general'
    
    def _personalize_response(self, response: str, context: Dict) -> str:
        """Personalize response based on context"""
        if context.get('message_count', 0) > 10:
            response = response.replace("I hear", "I continue to hear")
        
        if context.get('risk_trend') == 'escalating':
            response += " I notice things seem to be getting more difficult."
        elif context.get('risk_trend') == 'decreasing':
            response += " It sounds like you're working through this."
        
        return response
    
    def _get_follow_up_question(self, category: str) -> str:
        """Get appropriate follow-up question"""
        questions = {
            'anxiety': "What situations trigger these anxious feelings?",
            'depression': "How long have you been feeling this way?",
            'anger': "What do you think is at the root of this frustration?",
            'loneliness': "Tell me more about your support system.",
            'stress': "What's the biggest source of stress right now?",
            'general': "How can I best support you today?"
        }
        return questions.get(category, "Tell me more about what's on your mind.")
    
    def generate_ai_response(self, message: str, context: str) -> str:
        """Future: Integration with ChatGPT API"""
        # This is where you'll add ChatGPT integration
        prompt = f"""You are SafeMind, a compassionate mental health support assistant.
        
        Context: {context}
        User message: {message}
        
        Provide empathetic, supportive response. Focus on active listening.
        Do not provide medical advice. Encourage professional help when appropriate.
        """
        
        # Placeholder for API call
        # response = openai.ChatCompletion.create(...)
        
        return "AI response would go here"
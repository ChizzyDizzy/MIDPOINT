import json
import random
from typing import Dict

class CulturalAdapter:
    def __init__(self):
        with open('../data/cultural_templates.json', 'r') as f:
            self.templates = json.load(f)

    # Keywords that indicate spiritual/mindfulness suggestions are contextually relevant
    SPIRITUAL_CONTEXT_KEYWORDS = [
        'stress', 'stressed', 'anxious', 'anxiety', 'overwhelmed',
        'peace', 'calm', 'relax', 'cope', 'coping',
        'meditat', 'pray', 'spiritual', 'faith', 'hope',
        'sleep', 'restless', 'tension', 'breathe', 'breathing',
        'inner', 'soul', 'healing', 'mindful'
    ]

    def adapt_response(self, response: str, culture: str = 'south_asian') -> str:
        """Adapt response based on cultural context"""
        if culture not in self.templates:
            return response

        cultural_context = self.templates[culture]

        # Add greeting if at conversation start
        if cultural_context.get('use_formal_greeting'):
            response = f"{cultural_context['greeting']} {response}"

        # Add family-oriented suggestions only when contextually relevant
        if cultural_context.get('family_emphasis'):
            if 'support' in response.lower():
                response += f"\n{cultural_context['family_support_text']}"

        # Include spiritual elements ONLY when contextually appropriate
        if cultural_context.get('include_spiritual'):
            response_lower = response.lower()
            if any(kw in response_lower for kw in self.SPIRITUAL_CONTEXT_KEYWORDS):
                suggestions = cultural_context.get('spiritual_suggestions', [])
                if suggestions:
                    spiritual_suggestion = random.choice(suggestions)
                    response += f"\n{spiritual_suggestion}"

        return response
    
    def get_culturally_appropriate_resources(self, culture: str = 'south_asian') -> Dict:
        """Get culture-specific resources"""
        if culture in self.templates:
            return self.templates[culture].get('resources', {})
        return {}
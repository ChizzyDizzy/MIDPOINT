import json
from typing import Dict

class CulturalAdapter:
    def __init__(self):
        with open('../data/cultural_templates.json', 'r') as f:
            self.templates = json.load(f)
    
    def adapt_response(self, response: str, culture: str = 'south_asian') -> str:
        """Adapt response based on cultural context"""
        if culture not in self.templates:
            return response
        
        cultural_context = self.templates[culture]
        
        # Add greeting if at conversation start
        if cultural_context.get('use_formal_greeting'):
            response = f"{cultural_context['greeting']} {response}"
        
        # Add family-oriented suggestions
        if cultural_context.get('family_emphasis'):
            if 'support' in response.lower():
                response += f"\n{cultural_context['family_support_text']}"
        
        # Include spiritual elements if appropriate
        if cultural_context.get('include_spiritual'):
            spiritual_suggestion = cultural_context['spiritual_suggestions'][0]
            response += f"\n{spiritual_suggestion}"
        
        return response
    
    def get_culturally_appropriate_resources(self, culture: str = 'south_asian') -> Dict:
        """Get culture-specific resources"""
        if culture in self.templates:
            return self.templates[culture].get('resources', {})
        return {}
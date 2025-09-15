import re
import json
from typing import Dict, Tuple, List

class SafetyDetector:
    def __init__(self):
        # Load crisis patterns
        with open('../data/crisis_patterns.json', 'r') as f:
            self.crisis_data = json.load(f)
        
        self.crisis_keywords = self.crisis_data['keywords']
        self.crisis_phrases = self.crisis_data['phrases']
        
        # Compile regex patterns for efficiency
        self.crisis_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.crisis_data['patterns']
        ]
    
    def detect_crisis(self, text: str) -> Dict:
        """
        Detect crisis level in user message
        Returns: risk_level, confidence, triggers
        """
        text_lower = text.lower()
        triggers = []
        risk_scores = []
        
        # Check immediate risk keywords
        for keyword in self.crisis_keywords['immediate']:
            if keyword in text_lower:
                triggers.append(keyword)
                risk_scores.append(1.0)
        
        # Check high risk keywords
        for keyword in self.crisis_keywords['high']:
            if keyword in text_lower:
                triggers.append(keyword)
                risk_scores.append(0.8)
        
        # Check medium risk keywords
        for keyword in self.crisis_keywords['medium']:
            if keyword in text_lower:
                triggers.append(keyword)
                risk_scores.append(0.5)
        
        # Check patterns
        for pattern in self.crisis_patterns:
            if pattern.search(text):
                triggers.append('pattern_match')
                risk_scores.append(0.7)
        
        # Calculate overall risk
        if risk_scores:
            max_risk = max(risk_scores)
            avg_risk = sum(risk_scores) / len(risk_scores)
            
            if max_risk >= 0.9:
                risk_level = 'immediate'
            elif max_risk >= 0.7:
                risk_level = 'high'
            elif max_risk >= 0.5:
                risk_level = 'medium'
            else:
                risk_level = 'low'
        else:
            risk_level = 'none'
            max_risk = 0
            avg_risk = 0
        
        return {
            'risk_level': risk_level,
            'confidence': max_risk,
            'triggers': triggers,
            'requires_intervention': risk_level in ['immediate', 'high']
        }
    
    def generate_safety_response(self, risk_level: str) -> str:
        """Generate appropriate safety response based on risk level"""
        responses = {
            'immediate': """I'm very concerned about what you're sharing. Your safety is my top priority. 
                         Please reach out to a crisis helpline right now:
                         • National Crisis Line: 1333
                         • Emergency Services: 119
                         You don't have to go through this alone.""",
            
            'high': """I can hear that you're going through an incredibly difficult time. 
                    These feelings are serious, and you deserve support. Would you consider:
                    • Calling a counselor at 1926
                    • Talking to someone you trust
                    • Visiting your nearest hospital if you feel unsafe""",
            
            'medium': """It sounds like you're dealing with some heavy feelings. 
                      That takes courage to share. Remember that help is available:
                      • Mental Health Helpline: 1926
                      • You can also speak with a counselor or trusted friend""",
            
            'low': """Thank you for sharing how you're feeling. It's important to talk about 
                   these emotions. I'm here to listen and support you.""",
            
            'none': ""
        }
        return responses.get(risk_level, "")
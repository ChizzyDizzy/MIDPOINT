"""
SafeMind AI - Enhanced Safety Detector
Multi-layered crisis detection system with 9 detection layers
"""

import re
import json
from typing import Dict, List, Tuple
from textblob import TextBlob
import numpy as np

class EnhancedSafetyDetector:
    def __init__(self):
        """Initialize enhanced safety detection system"""
        # Load comprehensive crisis patterns
        try:
            with open('../data/enhanced_crisis_patterns.json', 'r') as f:
                self.crisis_data = json.load(f)
        except Exception:
            # Fallback to basic patterns
            with open('../data/crisis_patterns.json', 'r') as f:
                self.crisis_data = json.load(f)

        self.crisis_keywords = self.crisis_data['keywords']
        self.crisis_phrases = self.crisis_data.get('phrases', [])

        # Compile regex patterns for efficiency
        self.crisis_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.crisis_data.get('patterns', [])
        ]

        # Risk weights for multi-layered detection
        self.risk_weights = {
            'immediate_keywords': 1.0,
            'high_keywords': 1.0,
            'medium_keywords': 0.65,
            'low_keywords': 0.35,
            'pattern_match': 0.85,
            'sentiment_negative': 0.3,
            'contextual_indicators': 0.75,
            'temporal_urgency': 0.95,
            'planning_indicators': 1.0,
            'means_access': 1.0,
            'cultural_pressure': 0.35,
        }

    def detect_crisis(self, text: str, context: Dict = None) -> Dict:
        """
        Multi-layered crisis detection

        Args:
            text: User message to analyze
            context: Conversation context (optional)

        Returns:
            Dict with risk_level, confidence, triggers, and intervention details
        """
        text_lower = text.lower()
        triggers = []
        risk_scores = []
        detection_layers = {}

        # Layer 1: Immediate risk keywords (highest priority)
        immediate_score = self._check_immediate_risk(text_lower, triggers)
        if immediate_score > 0:
            risk_scores.append(('immediate_keywords', immediate_score))
            detection_layers['immediate_keywords'] = True

        # Layer 2: High risk keywords
        high_score = self._check_high_risk(text_lower, triggers)
        if high_score > 0:
            risk_scores.append(('high_keywords', high_score))
            detection_layers['high_keywords'] = True

        # Layer 3: Medium risk keywords
        medium_score = self._check_medium_risk(text_lower, triggers)
        if medium_score > 0:
            risk_scores.append(('medium_keywords', medium_score))
            detection_layers['medium_keywords'] = True

        # Layer 4: Low risk keywords
        low_score = self._check_low_risk(text_lower, triggers)
        if low_score > 0:
            risk_scores.append(('low_keywords', low_score))
            detection_layers['low_keywords'] = True

        # Layer 5: Pattern matching (suicide ideation, self-harm patterns)
        pattern_score = self._check_patterns(text, triggers)
        if pattern_score > 0:
            risk_scores.append(('pattern_match', pattern_score))
            detection_layers['pattern_match'] = True

        # Layer 6: Sentiment analysis
        sentiment_score = self._analyze_sentiment(text)
        if sentiment_score < -0.5:  # Highly negative
            risk_scores.append(('sentiment_negative', abs(sentiment_score)))
            detection_layers['sentiment_analysis'] = sentiment_score

        # Layer 7: Contextual indicators (hopelessness, isolation, burden)
        contextual_score = self._check_contextual_indicators(text_lower, triggers)
        if contextual_score > 0:
            risk_scores.append(('contextual_indicators', contextual_score))
            detection_layers['contextual_indicators'] = True

        # Layer 8: Temporal urgency (tonight, now, today)
        temporal_score = self._check_temporal_urgency(text_lower, triggers)
        if temporal_score > 0:
            risk_scores.append(('temporal_urgency', temporal_score))
            detection_layers['temporal_urgency'] = True

        # Layer 9: Planning indicators
        planning_score = self._check_planning_indicators(text_lower, triggers)
        if planning_score > 0:
            risk_scores.append(('planning_indicators', planning_score))
            detection_layers['planning_indicators'] = True

        # Layer 10: Means access indicators
        means_score = self._check_means_access(text_lower, triggers)
        if means_score > 0:
            risk_scores.append(('means_access', means_score))
            detection_layers['means_access'] = True

        # Layer 11: Cultural pressure indicators
        cultural_score = self._check_cultural_pressure(text_lower, triggers)
        if cultural_score > 0:
            risk_scores.append(('cultural_pressure', cultural_score))
            detection_layers['cultural_pressure'] = True

        # Calculate weighted risk score
        if risk_scores:
            weighted_scores = [score[1] * self.risk_weights[score[0]] for score in risk_scores]
            max_risk = max(weighted_scores)

            # Determine risk level
            if max_risk >= 0.9 or 'means_access' in detection_layers:
                risk_level = 'immediate'
                confidence = max_risk
            elif max_risk >= 0.7 or 'planning_indicators' in detection_layers:
                risk_level = 'high'
                confidence = max_risk
            elif max_risk >= 0.45:
                risk_level = 'medium'
                confidence = max_risk
            elif max_risk >= 0.2:
                risk_level = 'low'
                confidence = max_risk
            else:
                risk_level = 'minimal'
                confidence = max_risk
        else:
            risk_level = 'none'
            max_risk = 0
            confidence = 0

        # Enhanced intervention requirement logic
        requires_intervention = risk_level in ['immediate', 'high']

        return {
            'risk_level': risk_level,
            'confidence': round(confidence, 3),
            'triggers': list(set(triggers)),
            'requires_intervention': requires_intervention,
        }

    def _check_immediate_risk(self, text: str, triggers: List) -> float:
        """Check for immediate crisis keywords"""
        score = 0
        for keyword in self.crisis_keywords.get('immediate', []):
            if keyword in text:
                triggers.append(keyword)
                score = 1.0
        return score

    def _check_high_risk(self, text: str, triggers: List) -> float:
        """Check for high risk keywords"""
        score = 0
        for keyword in self.crisis_keywords.get('high', []):
            if keyword in text:
                triggers.append(keyword)
                score = max(score, 0.8)
        return score

    def _check_medium_risk(self, text: str, triggers: List) -> float:
        """Check for medium risk keywords"""
        score = 0
        for keyword in self.crisis_keywords.get('medium', []):
            if keyword in text:
                triggers.append(keyword)
                score = max(score, 0.7)
        return score

    def _check_low_risk(self, text: str, triggers: List) -> float:
        """Check for low risk keywords"""
        score = 0
        for keyword in self.crisis_keywords.get('low', []):
            if keyword in text:
                triggers.append(keyword)
                score = max(score, 0.9)
        return score

    def _check_patterns(self, text: str, triggers: List) -> float:
        """Check for crisis patterns using regex"""
        score = 0
        for pattern in self.crisis_patterns:
            if pattern.search(text):
                triggers.append('pattern_match')
                score = max(score, 0.85)
        return score

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of message"""
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def _check_contextual_indicators(self, text: str, triggers: List) -> float:
        """Check for contextual crisis indicators"""
        score = 0
        indicators = self.crisis_data.get('contextual_indicators', {})

        for indicator_type, keywords in indicators.items():
            for keyword in keywords:
                if keyword in text:
                    triggers.append(f"contextual_{indicator_type}")
                    score = max(score, 0.8)

        return score

    def _check_temporal_urgency(self, text: str, triggers: List) -> float:
        """Check for temporal urgency indicators"""
        score = 0
        escalation = self.crisis_data.get('risk_escalation_markers', {})
        temporal_markers = escalation.get('temporal', [])

        for marker in temporal_markers:
            if marker in text:
                triggers.append(f"urgent_{marker}")
                score = 0.95

        return score

    def _check_planning_indicators(self, text: str, triggers: List) -> float:
        """Check for planning/preparation indicators"""
        score = 0
        escalation = self.crisis_data.get('risk_escalation_markers', {})
        planning_markers = escalation.get('planning', [])

        for marker in planning_markers:
            if marker in text:
                triggers.append(f"planning_{marker}")
                score = 1.0

        return score

    def _check_means_access(self, text: str, triggers: List) -> float:
        """Check for access to lethal means"""
        score = 0
        escalation = self.crisis_data.get('risk_escalation_markers', {})
        means_markers = escalation.get('means', [])

        for marker in means_markers:
            if marker in text:
                triggers.append(f"means_{marker}")
                score = 1.0

        return score

    def _check_cultural_pressure(self, text: str, triggers: List) -> float:
        """Check for cultural pressure indicators"""
        score = 0
        cultural = self.crisis_data.get('cultural_considerations', {})

        for culture_key, culture_data in cultural.items():
            if isinstance(culture_data, dict):
                for indicator_type, keywords in culture_data.items():
                    for keyword in keywords:
                        if keyword in text:
                            triggers.append(f"cultural_{indicator_type}")
                            score = max(score, 0.9)

        return score

    def generate_safety_response(self, risk_level: str) -> str:
        """Generate appropriate safety response based on risk level"""
        responses = {
            'immediate': """I'm very concerned about your safety. Please reach out for help right now:

- National Crisis Hotline: 1333 (Available 24/7)
- Emergency Services: 119
- Sumithrayo (Emotional Support): 011-2696666

If you are in immediate danger, please go to your nearest hospital emergency room or call 119.

You are not alone. Help is available.""",

            'high': """What you're sharing is very serious. Your safety is important. Please consider reaching out:

- Mental Health Helpline: 1926
- Sumithrayo: 011-2696666
- Crisis Hotline: 1333

You can also talk to a trusted family member, friend, doctor, or counselor.
If you feel unsafe, go to the nearest hospital or call emergency services: 119.

You deserve support and care.""",

            'medium': """I hear that you're struggling. These feelings are important to address.

Resources available to you:
- Mental Health Helpline: 1926
- Sumithrayo (24/7): 011-2696666
- Professional counseling services

You can also talk to someone you trust or reach out to a mental health professional.

Remember: It's okay to ask for help.""",

            'low': """I appreciate you sharing how you're feeling. These emotions are valid.

Support is available if you need it:
- Mental Health Helpline: 1926
- Counseling services
- Support groups

Consider talking to someone you trust or a mental health professional if these feelings persist.""",

            'minimal': """Thank you for sharing. I'm here to listen and support you.

If you ever feel overwhelmed, remember:
- Mental Health Helpline: 1926
- Professional support is available""",

            'none': ""
        }

        return responses.get(risk_level, "")

"""
SafeMind AI - Enhanced Safety Detector
Multi-layered crisis detection system with ML-based approach
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
        except:
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
            'high_keywords': 0.8,
            'medium_keywords': 0.5,
            'pattern_match': 0.7,
            'sentiment_negative': 0.3,
            'contextual_indicators': 0.6,
            'temporal_urgency': 0.9,
            'planning_indicators': 0.95,
            'means_access': 1.0
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

        # Layer 4: Pattern matching (suicide ideation, self-harm patterns)
        pattern_score = self._check_patterns(text, triggers)
        if pattern_score > 0:
            risk_scores.append(('pattern_match', pattern_score))
            detection_layers['pattern_match'] = True

        # Layer 5: Sentiment analysis
        sentiment_score = self._analyze_sentiment(text)
        if sentiment_score < -0.5:  # Highly negative
            risk_scores.append(('sentiment_negative', abs(sentiment_score)))
            detection_layers['sentiment_analysis'] = sentiment_score

        # Layer 6: Contextual indicators (hopelessness, isolation, burden)
        contextual_score = self._check_contextual_indicators(text_lower, triggers)
        if contextual_score > 0:
            risk_scores.append(('contextual_indicators', contextual_score))
            detection_layers['contextual_indicators'] = True

        # Layer 7: Temporal urgency (tonight, now, today)
        temporal_score = self._check_temporal_urgency(text_lower, triggers)
        if temporal_score > 0:
            risk_scores.append(('temporal_urgency', temporal_score))
            detection_layers['temporal_urgency'] = True

        # Layer 8: Planning indicators
        planning_score = self._check_planning_indicators(text_lower, triggers)
        if planning_score > 0:
            risk_scores.append(('planning_indicators', planning_score))
            detection_layers['planning_indicators'] = True

        # Layer 9: Means access indicators
        means_score = self._check_means_access(text_lower, triggers)
        if means_score > 0:
            risk_scores.append(('means_access', means_score))
            detection_layers['means_access'] = True

        # Calculate weighted risk score
        if risk_scores:
            max_risk = max([score[1] * self.risk_weights[score[0]] for score in risk_scores])
            avg_risk = np.mean([score[1] * self.risk_weights[score[0]] for score in risk_scores])

            # Determine risk level with enhanced thresholds
            if max_risk >= 0.95 or 'means_access' in detection_layers:
                risk_level = 'immediate'
                confidence = max_risk
            elif max_risk >= 0.75 or 'planning_indicators' in detection_layers:
                risk_level = 'high'
                confidence = max_risk
            elif max_risk >= 0.5 or 'temporal_urgency' in detection_layers:
                risk_level = 'medium'
                confidence = avg_risk
            elif max_risk >= 0.3:
                risk_level = 'low'
                confidence = avg_risk
            else:
                risk_level = 'minimal'
                confidence = avg_risk
        else:
            risk_level = 'none'
            max_risk = 0
            avg_risk = 0
            confidence = 0

        # Enhanced intervention requirement logic
        requires_intervention = risk_level in ['immediate', 'high']
        requires_monitoring = risk_level in ['medium', 'low']

        return {
            'risk_level': risk_level,
            'confidence': round(confidence, 3),
            'triggers': list(set(triggers)),
            'requires_intervention': requires_intervention,
            'requires_monitoring': requires_monitoring,
            'detection_layers': detection_layers,
            'risk_scores': dict(risk_scores),
            'assessment_type': 'multi_layered_ml'
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
                score = max(score, 0.5)
        return score

    def _check_patterns(self, text: str, triggers: List) -> float:
        """Check for crisis patterns using regex"""
        score = 0
        for pattern in self.crisis_patterns:
            if pattern.search(text):
                triggers.append('pattern_match')
                score = max(score, 0.7)
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
                    score = max(score, 0.6)

        return score

    def _check_temporal_urgency(self, text: str, triggers: List) -> float:
        """Check for temporal urgency indicators"""
        score = 0
        escalation = self.crisis_data.get('risk_escalation_markers', {})
        temporal_markers = escalation.get('temporal', [])

        for marker in temporal_markers:
            if marker in text:
                triggers.append(f"urgent_{marker}")
                score = 0.9

        return score

    def _check_planning_indicators(self, text: str, triggers: List) -> float:
        """Check for planning/preparation indicators"""
        score = 0
        escalation = self.crisis_data.get('risk_escalation_markers', {})
        planning_markers = escalation.get('planning', [])

        for marker in planning_markers:
            if marker in text:
                triggers.append(f"planning_{marker}")
                score = 0.95

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

    def generate_safety_response(self, risk_level: str) -> str:
        """Generate appropriate safety response based on risk level"""
        responses = {
            'immediate': """🚨 **IMMEDIATE CRISIS RESPONSE** 🚨

I'm very concerned about your safety. Please take these steps RIGHT NOW:

1. **Call Emergency Services: 119**
2. **National Crisis Hotline: 1333** (Available 24/7)
3. **Sumithrayo: 011-2696666** (24/7 Emotional Support)

If you are in immediate danger:
- Go to your nearest hospital emergency room
- Call emergency services (119)
- Reach out to someone you trust immediately

You are not alone. Help is available.""",

            'high': """⚠️ **URGENT SUPPORT NEEDED** ⚠️

What you're sharing is very serious. Your safety is important. Please:

1. **Talk to someone now:**
   - Mental Health Helpline: 1926
   - Sumithrayo: 011-2696666
   - Crisis Hotline: 1333

2. **Reach out to:**
   - A trusted family member or friend
   - Your doctor or counselor
   - A mental health professional

3. **If you feel unsafe:**
   - Go to the nearest hospital
   - Call emergency services: 119

You deserve support and care.""",

            'medium': """⚠️ **SUPPORT RECOMMENDED** ⚠️

I hear that you're struggling. These feelings are important to address:

**Resources Available:**
- Mental Health Helpline: 1926
- Sumithrayo (24/7): 011-2696666
- Professional counseling services

**You can also:**
- Talk to someone you trust
- Reach out to your doctor
- Contact a mental health professional

Remember: It's okay to ask for help.""",

            'low': """I appreciate you sharing how you're feeling. These emotions are valid.

**Support is available:**
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

    def assess_conversation_trend(self, messages: List[Dict]) -> str:
        """Assess risk trend over conversation history"""
        if len(messages) < 2:
            return 'stable'

        risk_levels = {'none': 0, 'minimal': 1, 'low': 2, 'medium': 3, 'high': 4, 'immediate': 5}
        recent_risks = [risk_levels.get(msg.get('risk_level', 'none'), 0) for msg in messages[-5:]]

        if len(recent_risks) < 2:
            return 'stable'

        # Calculate trend
        trend = recent_risks[-1] - recent_risks[0]

        if trend >= 2:
            return 'escalating'
        elif trend <= -2:
            return 'improving'
        else:
            return 'stable'

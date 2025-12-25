"""
SafeMind AI - AI Model Integration
Handles OpenAI GPT API integration for generating empathetic mental health responses
"""

import os
import json
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SafeMindAI:
    def __init__(self):
        """Initialize OpenAI client and system prompts"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'sk-proj-demo-key-replace-with-actual':
            print("WARNING: Using fallback mode - No valid OpenAI API key found")
            self.use_ai = False
        else:
            self.client = OpenAI(api_key=api_key)
            self.use_ai = True

        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', 500))
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', 0.7))

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

        # Load response templates for fallback
        with open('../data/response_templates.json', 'r') as f:
            self.templates = json.load(f)

    def _load_system_prompt(self) -> str:
        """Load the system prompt for SafeMind AI"""
        return """You are SafeMind, a compassionate and culturally-aware mental health support assistant.

Your role is to:
1. Provide empathetic, non-judgmental support
2. Use active listening techniques
3. Validate emotions and experiences
4. Be culturally sensitive, especially to South Asian contexts
5. NEVER provide medical diagnoses or treatment plans
6. Encourage professional help when appropriate
7. Recognize crisis situations and prioritize safety

Communication style:
- Warm, empathetic, and supportive
- Use "I" statements (e.g., "I hear that you're...")
- Ask open-ended questions
- Reflect feelings back to the user
- Avoid clinical jargon
- Be concise but thorough

Cultural considerations:
- Understand family-centric values in South Asian culture
- Recognize stigma around mental health
- Respect intergenerational dynamics
- Acknowledge cultural expectations while supporting individual wellbeing

Safety protocols:
- If user expresses suicidal ideation, provide immediate crisis resources
- For self-harm, eating disorders, or substance abuse, recommend professional help
- Always prioritize user safety over conversation flow

Remember: You are a supportive companion, not a replacement for professional mental health care."""

    def generate_response(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str = 'none',
        emotion: str = 'neutral'
    ) -> str:
        """
        Generate AI response to user message

        Args:
            user_message: The user's input message
            context_summary: Conversation context and history
            risk_level: Detected risk level (none, low, medium, high, immediate)
            emotion: Detected emotion state

        Returns:
            AI-generated response string
        """
        if not self.use_ai:
            return self._generate_fallback_response(user_message, emotion)

        try:
            # Construct messages for the API
            messages = self._construct_messages(
                user_message,
                context_summary,
                risk_level,
                emotion
            )

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                presence_penalty=0.6,
                frequency_penalty=0.3
            )

            # Extract response text
            ai_response = response.choices[0].message.content.strip()

            # Add safety disclaimer for high-risk situations
            if risk_level in ['high', 'immediate']:
                ai_response = self._add_crisis_response(ai_response, risk_level)

            return ai_response

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._generate_fallback_response(user_message, emotion)

    def _construct_messages(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str,
        emotion: str
    ) -> List[Dict[str, str]]:
        """Construct message array for OpenAI API"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add context if available
        if context_summary.get('recent_messages'):
            context_text = "Previous conversation:\n"
            for msg in context_summary['recent_messages'][-3:]:  # Last 3 messages
                context_text += f"User: {msg.get('user_message', '')}\n"
                context_text += f"Assistant: {msg.get('bot_response', '')}\n"

            messages.append({
                "role": "system",
                "content": f"Context: {context_text}\n\nDetected emotion: {emotion}\nRisk level: {risk_level}"
            })

        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })

        return messages

    def _add_crisis_response(self, response: str, risk_level: str) -> str:
        """Add crisis resources to response for high-risk situations"""
        crisis_resources = {
            'immediate': """

🚨 **Immediate Support Resources:**
- **National Crisis Hotline (Sri Lanka): 1333**
- **Emergency Services: 119**
- **Sumithrayo (Emotional Support): 011-2696666**

You don't have to face this alone. Please reach out to one of these services right now.""",

            'high': """

**Support Resources:**
- **Mental Health Helpline: 1926**
- **Sumithrayo: 011-2696666**
- **Emergency Services (if in danger): 119**

Your wellbeing matters. Please consider reaching out to professional support."""
        }

        return response + crisis_resources.get(risk_level, '')

    def _generate_fallback_response(self, message: str, emotion: str) -> str:
        """Generate template-based response when AI is not available"""
        import random
        from textblob import TextBlob

        # Analyze sentiment
        blob = TextBlob(message.lower())
        sentiment = blob.sentiment.polarity

        # Determine category
        categories = {
            'anxiety': ['anxious', 'anxiety', 'worried', 'nervous', 'panic'],
            'depression': ['sad', 'depressed', 'hopeless', 'down', 'empty'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'burden'],
            'loneliness': ['lonely', 'alone', 'isolated', 'nobody'],
            'anger': ['angry', 'frustrated', 'mad', 'annoyed']
        }

        category = 'general'
        for cat, keywords in categories.items():
            if any(keyword in message.lower() for keyword in keywords):
                category = cat
                break

        # Select response
        if category in self.templates:
            response = random.choice(self.templates[category])
        else:
            response = random.choice(self.templates.get('general', [
                "I hear you. It takes courage to share how you're feeling.",
                "Thank you for opening up. I'm here to listen and support you.",
                "It sounds like you're going through a difficult time. Tell me more."
            ]))

        return response

    def generate_follow_up_question(self, category: str, context: Dict) -> str:
        """Generate contextually appropriate follow-up question"""
        questions = {
            'anxiety': [
                "What situations tend to trigger these anxious feelings?",
                "How long have you been experiencing this anxiety?",
                "What helps you feel calmer when anxiety strikes?"
            ],
            'depression': [
                "How has this been affecting your daily life?",
                "Have you been able to talk to anyone about how you're feeling?",
                "What used to bring you joy before these feelings started?"
            ],
            'stress': [
                "What's the biggest source of stress for you right now?",
                "How have you been trying to cope with this stress?",
                "Is there anything that helps you feel less overwhelmed?"
            ],
            'loneliness': [
                "Tell me more about your social connections.",
                "What makes you feel most isolated?",
                "Are there activities or places where you feel less alone?"
            ],
            'general': [
                "What's on your mind today?",
                "How can I best support you?",
                "Tell me more about what you're experiencing."
            ]
        }

        import random
        return random.choice(questions.get(category, questions['general']))

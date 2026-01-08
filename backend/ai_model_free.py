"""
SafeMind AI - Hugging Face Model Integration
Supports Hugging Face models only:
1. Hugging Face Inference API (FREE tier - recommended)
2. Local Hugging Face models (completely FREE, runs offline)
"""

import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import requests

load_dotenv()

class SafeMindAI:
    def __init__(self):
        """Initialize AI client with Hugging Face"""

        # Determine which AI backend to use
        self.ai_backend = os.getenv('AI_BACKEND', 'huggingface')  # huggingface, local, or fallback

        # Initialize based on backend
        if self.ai_backend == 'huggingface':
            self._init_huggingface()
        elif self.ai_backend == 'local':
            self._init_local_model()
        else:
            print("WARNING: Using fallback mode - template-based responses")
            self.use_ai = False

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

        # Load response templates for fallback
        with open('../data/response_templates.json', 'r') as f:
            self.templates = json.load(f)

    def _init_huggingface(self):
        """Initialize Hugging Face Inference API (FREE tier)"""
        api_key = os.getenv('HUGGINGFACE_API_KEY')

        if not api_key or api_key.startswith('hf_your'):
            print("WARNING: No Hugging Face API key found. Using fallback mode.")
            print("Get a FREE API key at: https://huggingface.co/settings/tokens")
            self.use_ai = False
            return

        self.hf_api_key = api_key
        self.hf_model = os.getenv('HUGGINGFACE_MODEL', 'microsoft/DialoGPT-medium')
        # Alternative mental health models:
        # - 'facebook/blenderbot-400M-distill' (conversational)
        # - 'microsoft/DialoGPT-medium' (dialogue)
        # - 'google/flan-t5-base' (instruction following)
        # - 'distilgpt2' (lightweight and fast)

        self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        self.use_ai = True
        print(f"✓ Hugging Face initialized with model: {self.hf_model}")

    def _init_local_model(self):
        """Initialize local Hugging Face model (completely FREE, offline)"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = os.getenv('LOCAL_MODEL', 'microsoft/DialoGPT-small')
            print(f"Loading local model: {model_name}")
            print("First time will download ~500MB, then runs offline...")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.local_model.to(self.device)

            self.use_ai = True
            print(f"✓ Local model loaded on {self.device}")

        except ImportError:
            print("WARNING: transformers not installed. Run: pip install transformers torch")
            self.use_ai = False
        except Exception as e:
            print(f"WARNING: Failed to load local model: {e}")
            self.use_ai = False

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
        Generate AI response to user message using Hugging Face models

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
            # Route to appropriate backend
            if self.ai_backend == 'huggingface':
                ai_response = self._generate_huggingface_response(
                    user_message, context_summary, risk_level, emotion
                )
            elif self.ai_backend == 'local':
                ai_response = self._generate_local_response(
                    user_message, context_summary, risk_level, emotion
                )
            else:
                ai_response = self._generate_fallback_response(user_message, emotion)

            # Add safety disclaimer for high-risk situations
            if risk_level in ['high', 'immediate']:
                ai_response = self._add_crisis_response(ai_response, risk_level)

            return ai_response

        except Exception as e:
            print(f"Error generating AI response: {e}")
            return self._generate_fallback_response(user_message, emotion)

    def _generate_huggingface_response(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str,
        emotion: str
    ) -> str:
        """Generate response using Hugging Face Inference API (FREE)"""

        # Construct prompt with context
        prompt = self._construct_prompt(user_message, context_summary, risk_level, emotion)

        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "return_full_text": False
            }
        }

        response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            elif isinstance(result, dict):
                return result.get('generated_text', '').strip()
        elif response.status_code == 503:
            print("Model is loading... This may take 20-30 seconds on first use.")
            # Retry after model loads
            import time
            time.sleep(20)
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    return result[0].get('generated_text', '').strip()

        # Fallback if API fails
        return self._generate_fallback_response(user_message, emotion)

    def _generate_local_response(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str,
        emotion: str
    ) -> str:
        """Generate response using local Hugging Face model (completely FREE)"""
        import torch

        # Encode input
        input_text = f"User: {user_message}\nBot:"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        # Generate response
        with torch.no_grad():
            output = self.local_model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract only the bot's response
        if "Bot:" in response:
            response = response.split("Bot:")[-1].strip()

        return response

    def _construct_prompt(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str,
        emotion: str
    ) -> str:
        """Construct prompt for AI models"""

        prompt = f"{self.system_prompt}\n\n"

        # Add context if available
        if context_summary.get('recent_messages'):
            prompt += "Previous conversation:\n"
            for msg in context_summary['recent_messages'][-2:]:  # Last 2 messages
                prompt += f"User: {msg.get('user_message', '')}\n"
                prompt += f"Assistant: {msg.get('bot_response', '')}\n"
            prompt += "\n"

        prompt += f"User emotion: {emotion}\n"
        prompt += f"Risk level: {risk_level}\n\n"
        prompt += f"User: {user_message}\n"
        prompt += "Assistant:"

        return prompt

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

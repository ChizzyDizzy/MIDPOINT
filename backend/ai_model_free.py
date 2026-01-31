"""
SafeMind AI - AI Model Integration
Supports multiple backends:
1. OpenAI API (ChatGPT - recommended for best responses)
2. Hugging Face Inference API (FREE tier)
3. Local Hugging Face models (offline)
4. Template-based fallback (no API needed)
"""

import os
import json
import random
from typing import Dict, List, Optional
from dotenv import load_dotenv
import requests

load_dotenv()

class SafeMindAI:
    def __init__(self):
        """Initialize AI client"""

        # Determine which AI backend to use
        self.ai_backend = os.getenv('AI_BACKEND', 'openai')

        # Initialize based on backend
        if self.ai_backend == 'openai':
            self._init_openai()
        elif self.ai_backend == 'huggingface':
            self._init_huggingface()
        elif self.ai_backend == 'local':
            self._init_local_model()
        else:
            print("Using fallback mode - template-based responses")
            self.use_ai = False

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

        # Load response templates for fallback
        try:
            with open('../data/response_templates.json', 'r') as f:
                self.templates = json.load(f)
        except Exception:
            self.templates = {}

    def _init_openai(self):
        """Initialize OpenAI API (ChatGPT)"""
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')

        if not self.openai_api_key or self.openai_api_key.startswith('sk-your'):
            print("WARNING: No OpenAI API key found. Using fallback mode.")
            print("Get an API key at: https://platform.openai.com/api-keys")
            print("Then set OPENAI_API_KEY in your .env file")
            self.use_ai = False
            return

        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.use_ai = True
        print(f"OpenAI initialized with model: {self.openai_model}")

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
        self.hf_api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        self.use_ai = True
        print(f"Hugging Face initialized with model: {self.hf_model}")

    def _init_local_model(self):
        """Initialize local Hugging Face model (completely FREE, offline)"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = os.getenv('LOCAL_MODEL', os.getenv('HUGGINGFACE_MODEL', 'microsoft/DialoGPT-small'))
            print(f"Loading local model: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.local_model.to(self.device)

            self.use_ai = True
            print(f"Local model loaded on {self.device}")

        except ImportError:
            print("WARNING: transformers not installed. Run: pip install transformers torch")
            self.use_ai = False
        except Exception as e:
            print(f"WARNING: Failed to load local model: {e}")
            self.use_ai = False

    def _load_system_prompt(self) -> str:
        """Load the system prompt for SafeMind AI"""
        return """You are SafeMind, a compassionate and culturally-aware mental health support chatbot designed specifically for people in Sri Lanka.

Your role:
1. Provide empathetic, non-judgmental emotional support
2. Use active listening - reflect feelings back to the user
3. Be culturally sensitive to Sri Lankan and South Asian contexts
4. NEVER diagnose conditions or prescribe medication
5. Recognize crisis situations and provide Sri Lankan emergency numbers
6. Encourage professional help when appropriate

Cultural context (Sri Lanka):
- Understand family-centric values and respect for elders
- Recognize stigma around mental health in Sri Lankan society
- Be aware of academic pressure (O/L, A/L exams, university entrance)
- Understand arranged marriage expectations and family duty
- Recognize intergenerational dynamics and parental expectations
- Be sensitive to economic pressures and job market stress
- Understand the role of religion (Buddhism, Hinduism, Islam, Christianity) as coping
- Know that "what will people say" (social stigma) is a major barrier to seeking help

Communication style:
- Warm, empathetic, and supportive
- Use "I" statements ("I hear that you're...", "I can sense that...")
- Ask open-ended questions to encourage sharing
- Keep responses concise (2-4 sentences typically)
- Don't be preachy or lecture the user
- Be genuine and conversational, not robotic

Sri Lankan emergency resources (mention only when needed):
- Crisis Hotline: 1333 (24/7)
- Mental Health Helpline: 1926
- Emergency Services: 119
- Sumithrayo (Emotional Support): 011-2696666

Remember: You are a supportive companion, NOT a replacement for professional mental health care."""

    def generate_response(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str = 'none',
        emotion: str = 'neutral'
    ) -> str:
        """Generate AI response to user message"""
        if not self.use_ai:
            return self._generate_fallback_response(user_message, emotion)

        try:
            if self.ai_backend == 'openai':
                ai_response = self._generate_openai_response(
                    user_message, context_summary, risk_level, emotion
                )
            elif self.ai_backend == 'huggingface':
                ai_response = self._generate_huggingface_response(
                    user_message, context_summary, risk_level, emotion
                )
            elif self.ai_backend == 'local':
                ai_response = self._generate_local_response(
                    user_message, context_summary, risk_level, emotion
                )
            else:
                ai_response = self._generate_fallback_response(user_message, emotion)

            # Validate response quality
            if self._is_garbage_response(ai_response):
                ai_response = self._generate_fallback_response(user_message, emotion)

            # Add safety resources for high-risk situations
            if risk_level in ['high', 'immediate']:
                ai_response = self._add_crisis_response(ai_response, risk_level)

            return ai_response

        except Exception as e:
            print(f"Error generating AI response: {e}")
            return self._generate_fallback_response(user_message, emotion)

    def _generate_openai_response(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str,
        emotion: str
    ) -> str:
        """Generate response using OpenAI ChatGPT API"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add risk context if elevated
        if risk_level not in ['none', 'minimal']:
            messages.append({
                "role": "system",
                "content": f"The user's message has been assessed as {risk_level} risk. "
                           f"Respond with appropriate care and sensitivity for this risk level."
            })

        # Add conversation history
        if context_summary.get('recent_messages'):
            for msg in context_summary['recent_messages'][-3:]:
                messages.append({"role": "user", "content": msg.get('user_message', '')})
                messages.append({"role": "assistant", "content": msg.get('bot_response', '')})

        messages.append({"role": "user", "content": user_message})

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.openai_model,
            "messages": messages,
            "max_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        elif response.status_code == 401:
            print("OpenAI API key is invalid. Check your OPENAI_API_KEY in .env")
        elif response.status_code == 429:
            print("OpenAI rate limit reached. Using fallback.")
        else:
            print(f"OpenAI API error {response.status_code}: {response.text[:200]}")

        return self._generate_fallback_response(user_message, emotion)

    def _generate_huggingface_response(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str,
        emotion: str
    ) -> str:
        """Generate response using Hugging Face Inference API"""
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
            print("Model is loading... retrying in 20s")
            import time
            time.sleep(20)
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list):
                    return result[0].get('generated_text', '').strip()

        return self._generate_fallback_response(user_message, emotion)

    def _generate_local_response(
        self,
        user_message: str,
        context_summary: Dict,
        risk_level: str,
        emotion: str
    ) -> str:
        """Generate response using local Hugging Face model"""
        import torch

        input_text = f"<|user|> {user_message}\n<|assistant|>"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            output = self.local_model.generate(
                input_ids,
                max_new_tokens=150,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        elif "<|user|>" in response:
            parts = response.split("<|user|>")
            response = parts[-1].strip() if len(parts) > 1 else response

        if self._is_garbage_response(response):
            return self._generate_fallback_response(user_message, emotion)

        return response

    def _is_garbage_response(self, response: str) -> bool:
        """Check if a model response is garbage"""
        if not response or len(response.strip()) < 10:
            return True

        garbage_indicators = [
            'u/', 'r/', '/u/', '/r/', 'reddit',
            'Bot', 'bot ', 'Poster', 'Cleaner',
            'http://', 'https://', 'www.',
            'lol', 'lmao', 'rofl', 'bruh',
            'User :', 'User:', 'Bot:',
            'neuron train', 'Eight 1911', 'themightygod',
        ]
        response_lower = response.lower()
        for indicator in garbage_indicators:
            if indicator.lower() in response_lower:
                return True

        special_count = sum(1 for c in response if not c.isalnum() and c not in ' .,!?\'"-:;()\n')
        if len(response) > 0 and special_count / len(response) > 0.3:
            return True

        return False

    def _construct_prompt(self, user_message, context_summary, risk_level, emotion):
        """Construct prompt for HuggingFace models"""
        prompt = f"{self.system_prompt}\n\n"
        if context_summary.get('recent_messages'):
            prompt += "Previous conversation:\n"
            for msg in context_summary['recent_messages'][-2:]:
                prompt += f"User: {msg.get('user_message', '')}\n"
                prompt += f"Assistant: {msg.get('bot_response', '')}\n"
            prompt += "\n"
        prompt += f"User: {user_message}\nAssistant:"
        return prompt

    def _add_crisis_response(self, response: str, risk_level: str) -> str:
        """Add crisis resources for high-risk situations"""
        crisis_resources = {
            'immediate': "\n\nImmediate Support:\n- Crisis Hotline (Sri Lanka): 1333\n- Emergency Services: 119\n- Sumithrayo: 011-2696666\n\nPlease reach out now. You don't have to face this alone.",
            'high': "\n\nSupport Resources:\n- Mental Health Helpline: 1926\n- Sumithrayo: 011-2696666\n- Emergency (if in danger): 119\n\nYour wellbeing matters. Please consider reaching out to professional support."
        }
        return response + crisis_resources.get(risk_level, '')

    def _generate_fallback_response(self, message: str, emotion: str) -> str:
        """Generate template-based response when AI is not available"""
        message_lower = message.lower()

        categories = {
            'crisis': ['suicide', 'kill myself', 'end my life', 'want to die', 'hurt myself', 'self-harm'],
            'anxiety': ['anxious', 'anxiety', 'worried', 'nervous', 'panic', 'scared'],
            'depression': ['sad', 'depressed', 'hopeless', 'down', 'empty', 'numb', 'crying'],
            'stress': ['stressed', 'overwhelmed', 'pressure', 'burden', 'stressful', 'overworked'],
            'academic_stress': ['exam', 'a/l', 'o/l', 'results', 'university', 'grades', 'study'],
            'family_issues': ['family', 'parents', 'mother', 'father', 'marriage', 'home'],
            'cultural_pressure': ['society', 'expect', 'shame', 'honor', 'tradition', 'duty'],
            'loneliness': ['lonely', 'alone', 'isolated', 'nobody', 'no friends'],
            'anger': ['angry', 'frustrated', 'mad', 'annoyed', 'furious'],
            'self_harm': ['cutting', 'hurt myself', 'harm myself', 'self-harm'],
            'positive': ['better', 'good', 'happy', 'helped', 'improving', 'thank'],
        }

        category = 'general'
        for cat, keywords in categories.items():
            if any(keyword in message_lower for keyword in keywords):
                category = cat
                break

        if category in self.templates and self.templates[category]:
            response = random.choice(self.templates[category])
        else:
            response = random.choice(self.templates.get('general', [
                "I hear you. It takes courage to share how you're feeling. What's been weighing on you most?",
                "Thank you for opening up. I'm here to listen and support you. Can you tell me more?",
                "It sounds like you're going through a difficult time. I'm here for you.",
            ]))

        return response

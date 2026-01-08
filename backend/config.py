import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True

    # Hugging Face configuration
    HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
    HUGGINGFACE_MODEL = os.environ.get('HUGGINGFACE_MODEL', 'microsoft/DialoGPT-medium')

    # Crisis hotlines for Sri Lanka
    EMERGENCY_RESOURCES = {
        'sri_lanka': {
            'suicide_hotline': '1333',
            'mental_health': '1926',
            'emergency': '119'
        }
    }

    # Safety thresholds
    CRISIS_THRESHOLD = 0.7
    WARNING_THRESHOLD = 0.5
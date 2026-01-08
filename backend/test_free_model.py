"""
Quick test script to verify FREE AI model is working
Run this to confirm your Hugging Face or local model setup
"""

import sys
import os

print("=" * 70)
print(" " * 15 + "SafeMind AI - FREE Model Test")
print("=" * 70)
print()

# Test 1: Check environment
print("[1/4] Checking environment configuration...")
print("-" * 70)

from dotenv import load_dotenv
load_dotenv()

ai_backend = os.getenv('AI_BACKEND', 'fallback')
print(f"✓ AI Backend: {ai_backend}")

if ai_backend == 'huggingface':
    api_key = os.getenv('HUGGINGFACE_API_KEY', '')
    model = os.getenv('HUGGINGFACE_MODEL', 'microsoft/DialoGPT-medium')

    if not api_key or api_key == 'hf_your_actual_key_here':
        print("❌ ERROR: No valid Hugging Face API key found!")
        print()
        print("To fix:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a FREE token")
        print("3. Add to .env file:")
        print("   HUGGINGFACE_API_KEY=hf_your_key_here")
        sys.exit(1)

    print(f"✓ Hugging Face API Key: {api_key[:10]}...")
    print(f"✓ Model: {model}")

elif ai_backend == 'local':
    model = os.getenv('LOCAL_MODEL', 'microsoft/DialoGPT-small')
    print(f"✓ Local Model: {model}")
    print("  (First run will download ~500MB)")

else:
    print("⚠️  Fallback mode (template responses, no AI)")

print()

# Test 2: Import AI model
print("[2/4] Loading AI model...")
print("-" * 70)

try:
    from ai_model_free import SafeMindAI
    print("✓ ai_model_free.py loaded successfully")
except ImportError as e:
    print(f"❌ ERROR: Could not import ai_model_free: {e}")
    print()
    print("Make sure ai_model_free.py exists in backend/ folder")
    sys.exit(1)

print()

# Test 3: Initialize AI
print("[3/4] Initializing AI model...")
print("-" * 70)

try:
    ai_model = SafeMindAI()

    if ai_model.use_ai:
        print("✓ AI model initialized successfully!")
        print(f"  Backend: {ai_model.ai_backend}")
    else:
        print("⚠️  AI initialized in fallback mode (templates only)")
        print("  Check your API key configuration")

except Exception as e:
    print(f"❌ ERROR initializing AI: {e}")
    sys.exit(1)

print()

# Test 4: Generate response
print("[4/4] Testing AI response generation...")
print("-" * 70)

test_message = "I've been feeling really anxious about my upcoming exams"
context = {'recent_messages': []}

print(f"Test input: \"{test_message}\"")
print()

if ai_backend == 'huggingface':
    print("⏳ Generating response via Hugging Face API...")
    print("   (First call may take 20-30 seconds - model loading)")
else:
    print("⏳ Generating response...")

print()

try:
    response = ai_model.generate_response(
        user_message=test_message,
        context_summary=context,
        risk_level='low',
        emotion='anxious'
    )

    print("✓ Response generated successfully!")
    print()
    print("AI Response:")
    print("─" * 70)
    print(response)
    print("─" * 70)
    print()

    # Check if it's actually AI or template
    if ai_model.use_ai and ai_backend != 'fallback':
        print("✅ SUCCESS! AI model is working!")
        print()
        print("Your system is ready for MVP demonstration!")
    else:
        print("⚠️  Using template responses (no AI)")
        print("   This will work but won't show AI capabilities")

except Exception as e:
    print(f"❌ ERROR generating response: {e}")
    print()

    if ai_backend == 'huggingface':
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify your Hugging Face API key is correct")
        print("3. Wait 30 seconds and try again (model loading)")
        print("4. Check https://status.huggingface.co/ for service status")

    sys.exit(1)

print()
print("=" * 70)
print(" " * 15 + "🎉 All Tests Passed! 🎉")
print("=" * 70)
print()
print("Next steps:")
print("1. Run the full application: python app_improved.py")
print("2. Run comprehensive tests: python test_mvp.py")
print("3. Start the frontend and test the UI")
print()
print("For your MVP demo:")
print("- Show this test output (proves AI is working)")
print("- Show the synthetic dataset (data/training_conversations.json)")
print("- Show the test results (test_mvp.py output)")
print("- Explain Input → Process → Output flow")
print()
print("=" * 70)

"""
List available Gemini models using simple requests
"""
import os
import requests
import json

# Get API key from environment
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    print("Run: export GEMINI_API_KEY=your_key")
    exit(1)

print(f"✓ Using API key: {api_key[:15]}...\n")
print("=" * 70)
print("Fetching Available Gemini Models...")
print("=" * 70)

# Try v1beta endpoint
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])

        print(f"\n✓ Found {len(models)} models\n")

        for model in models:
            name = model.get('name', 'Unknown')
            display = model.get('displayName', 'Unknown')
            methods = model.get('supportedGenerationMethods', [])
            description = model.get('description', '')

            print(f"📦 {name}")
            print(f"   Display: {display}")

            if description:
                print(f"   Description: {description[:80]}...")

            print(f"   Methods: {', '.join(methods)}")

            # Highlight if it supports generateContent
            if 'generateContent' in methods:
                print("   ✅ USABLE for dataset generation!")
            print()

        print("=" * 70)
        print("RECOMMENDED MODELS FOR FREE TIER:")
        print("=" * 70)

        for model in models:
            name = model.get('name', '')
            methods = model.get('supportedGenerationMethods', [])

            if 'generateContent' in methods and 'flash' in name.lower():
                print(f"✅ {name}")

    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 70)
print("Copy the exact model name (without 'models/' prefix) to use in code")
print("=" * 70)

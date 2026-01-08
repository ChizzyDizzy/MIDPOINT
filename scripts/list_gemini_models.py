"""
List available Gemini models to find the correct one to use
"""
import os
from google import genai

# Get API key from environment
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("ERROR: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
    exit(1)

print(f"✓ Using API key: {api_key[:10]}...\n")
print("=" * 70)
print("Available Gemini Models:")
print("=" * 70)

try:
    client = genai.Client(api_key=api_key)

    # List all available models
    models = client.models.list()

    print(f"\nFound {len(list(models))} models:\n")

    models = client.models.list()  # Call again to iterate
    for model in models:
        print(f"Model: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description[:100] if hasattr(model, 'description') and model.description else 'N/A'}...")

        # Check if it supports generateContent
        if hasattr(model, 'supported_generation_methods'):
            methods = model.supported_generation_methods
            print(f"  Supports generateContent: {'generateContent' in methods}")
        print()

except Exception as e:
    print(f"Error: {e}")
    print("\nTrying alternative method...")

    # If that doesn't work, try to just list model names
    try:
        import requests
        headers = {'Content-Type': 'application/json'}
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            print("\nAvailable models:")
            for model in data.get('models', []):
                name = model.get('name', 'Unknown')
                display = model.get('displayName', 'Unknown')
                methods = model.get('supportedGenerationMethods', [])

                print(f"\n{name}")
                print(f"  Display: {display}")
                print(f"  Methods: {', '.join(methods)}")

                # Highlight if it supports generateContent
                if 'generateContent' in methods:
                    print("  ✅ Can be used for dataset generation!")
        else:
            print(f"API Error: {response.status_code}")
            print(response.text)
    except Exception as e2:
        print(f"Alternative method also failed: {e2}")

print("\n" + "=" * 70)
print("Use one of the models above that supports 'generateContent'")
print("=" * 70)

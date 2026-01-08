"""
SafeMind AI - Synthetic Dataset Generator
Generates culturally-aware mental health training conversations

This script uses Gemini (FREE) to generate high-quality synthetic training
data for the Sri Lankan mental health context.

QUICK START WITH GEMINI (FREE):
1. Get FREE API key: https://aistudio.google.com/app/apikey
2. Set environment variable:
   export GEMINI_API_KEY=your-key-here
3. Run script (no extra packages needed!):
   python3 generate_dataset.py --provider gemini --num-samples 500 --output ../data/mental_health_dataset.json

USAGE:
  python3 generate_dataset.py --provider gemini --num-samples 500
  python3 generate_dataset.py --provider gemini --api-key YOUR_KEY --num-samples 1000

NOTE: Python 3.10+ recommended (you have 3.9, which works but shows warnings)
"""

import json
import os
import time
from typing import List, Dict, Optional
import random
import argparse

# Import API clients (install: pip install anthropic openai google-generativeai)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: anthropic not installed. Install with: pip install anthropic")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

# Gemini uses REST API directly - no special package needed
# Just need requests which is already a dependency
GEMINI_AVAILABLE = True  # Always available since we use REST API


class DatasetGenerator:
    """Generate synthetic mental health conversation dataset"""

    def __init__(self, api_key: str, provider: str = "claude"):
        """
        Initialize dataset generator

        Args:
            api_key: API key for LLM provider
            provider: "openai", "claude", or "gemini"
        """
        self.provider = provider.lower()
        self.api_key = api_key

        # Initialize API client
        if self.provider == "claude":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic library not installed")
            self.client = Anthropic(api_key=api_key)
            self.model = "claude-3-5-sonnet-20241022"

        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai library not installed")
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4"

        elif self.provider == "gemini":
            # Store API key for REST API calls
            self.api_key = api_key
            # Use gemini-1.5-flash-latest (FREE tier model)
            # We use REST API directly - more reliable than SDK
            self.model = "gemini-1.5-flash-latest"

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate_sample(self, user_input: str, category: str,
                       emotion: str, risk_level: str) -> Optional[Dict]:
        """
        Generate a single training sample

        Args:
            user_input: Example user message (seed)
            category: anxiety, depression, stress, crisis, etc.
            emotion: sad, anxious, hopeless, stressed, neutral
            risk_level: low, medium, high

        Returns:
            Dict with instruction, input, response, emotion, risk_level
        """

        prompt = f"""Generate a mental health chatbot training sample for the Sri Lankan context.

CONTEXT:
- User message category: {category}
- Emotion: {emotion}
- Risk level: {risk_level}
- Cultural context: Sri Lanka (family pressure, academic stress, social stigma)

USER MESSAGE (example):
"{user_input}"

Generate a SIMILAR but DIFFERENT user message and appropriate response.

REQUIREMENTS FOR RESPONSE:
1. DO NOT diagnose any mental illness
2. DO NOT provide medical or therapeutic advice
3. Use empathetic, supportive tone
4. Reflect Sri Lankan cultural context:
   - Family expectations and pressure
   - Academic stress (A/L exams, university)
   - Social stigma around mental health
   - Financial concerns
5. Encourage self-reflection
6. Suggest professional help gently when appropriate
7. Use Sri Lankan English (respectful, neutral)
8. If high risk: Include crisis resources (1333 hotline)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "instruction": "You are a mental health awareness chatbot for Sri Lankan users. Provide empathetic, culturally-aware support without diagnosing or replacing professional care.",
  "input": "[Generated user message - make it realistic and different from example]",
  "response": "[Your empathetic, culturally-appropriate response]",
  "emotion": "{emotion}",
  "risk_level": "{risk_level}"
}}"""

        try:
            if self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content

            elif self.provider == "gemini":
                # Use REST API directly for reliability
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 1024,
                    }
                }

                response = requests.post(url, json=payload, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    content = result['candidates'][0]['content']['parts'][0]['text']
                else:
                    raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

            # Parse JSON from response
            # Remove markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            sample = json.loads(content)

            # Validate required fields
            required_fields = ["instruction", "input", "response", "emotion", "risk_level"]
            if all(field in sample for field in required_fields):
                return sample
            else:
                print(f"Warning: Missing fields in generated sample")
                return None

        except Exception as e:
            print(f"Error generating sample: {e}")
            return None

    def generate_dataset(self, num_samples: int = 1000,
                        output_file: str = "../data/synthetic_training_data.json",
                        checkpoint_interval: int = 100) -> List[Dict]:
        """
        Generate complete dataset

        Args:
            num_samples: Number of samples to generate
            output_file: Where to save the dataset
            checkpoint_interval: Save checkpoint every N samples

        Returns:
            List of generated samples
        """

        # Seed examples for each category
        seeds = {
            "anxiety": [
                ("I'm very nervous about my A/L results", "anxious", "medium"),
                ("I can't sleep because I'm worried about the interview", "anxious", "medium"),
                ("My heart races when I think about exams", "anxious", "low"),
                ("I feel panicky all the time", "anxious", "medium"),
                ("I'm scared about what others think of me", "anxious", "low"),
            ],
            "depression": [
                ("I feel sad all the time and nothing makes me happy", "sad", "medium"),
                ("I have no energy to do anything anymore", "sad", "medium"),
                ("Everything feels meaningless lately", "sad", "high"),
                ("I don't enjoy things I used to love", "sad", "medium"),
                ("I feel empty inside", "sad", "high"),
            ],
            "family_pressure": [
                ("My parents want me to be a doctor but I want to study arts", "stressed", "low"),
                ("Everyone compares me to my cousin who went abroad", "sad", "medium"),
                ("I feel like I'm disappointing my family", "sad", "medium"),
                ("My family doesn't understand my choices", "stressed", "low"),
                ("There's too much pressure to get married", "stressed", "medium"),
            ],
            "academic_stress": [
                ("I failed my university exam and don't know what to do", "stressed", "medium"),
                ("There's too much pressure to get high marks", "anxious", "low"),
                ("I can't handle the workload at university", "stressed", "medium"),
                ("I'm afraid I'll fail my A/Levels", "anxious", "medium"),
                ("My grades are dropping and I feel hopeless", "sad", "high"),
            ],
            "financial_stress": [
                ("My family is struggling financially and I feel guilty", "stressed", "medium"),
                ("I can't find a job after graduation", "sad", "medium"),
                ("I worry about money all the time", "anxious", "low"),
                ("I can't afford university fees", "stressed", "medium"),
                ("The cost of living is overwhelming", "stressed", "low"),
            ],
            "crisis": [
                ("I don't see the point in living anymore", "hopeless", "high"),
                ("I feel completely hopeless about everything", "hopeless", "high"),
                ("I want everything to end", "hopeless", "high"),
                ("I feel like a burden to everyone", "hopeless", "high"),
                ("Nothing will ever get better", "hopeless", "high"),
            ],
            "relationship": [
                ("I feel very lonely and have no one to talk to", "sad", "medium"),
                ("My friends don't understand what I'm going through", "sad", "low"),
                ("I feel isolated from everyone", "sad", "medium"),
                ("I'm having problems in my relationship", "stressed", "low"),
                ("I feel like nobody cares about me", "sad", "medium"),
            ],
            "work_stress": [
                ("My boss puts too much pressure on me", "stressed", "medium"),
                ("I'm working long hours and feel burned out", "stressed", "medium"),
                ("I hate my job but can't quit", "sad", "low"),
                ("Work stress is affecting my health", "anxious", "medium"),
            ],
            "positive": [
                ("I tried the breathing exercises and they helped", "neutral", "low"),
                ("I'm feeling a bit better after talking", "neutral", "low"),
                ("Thank you for listening to me", "neutral", "low"),
                ("I think I'm making progress", "neutral", "low"),
                ("I feel more hopeful today", "neutral", "low"),
            ]
        }

        dataset = []
        categories = list(seeds.keys())

        print("=" * 60)
        print(f"Generating {num_samples} synthetic training samples...")
        print(f"Provider: {self.provider} | Model: {self.model}")
        print("=" * 60)

        for i in range(num_samples):
            # Select category (weighted to avoid too many crisis samples)
            if i % 20 == 0 and i > 0:  # 5% crisis samples
                category = "crisis"
            elif i % 15 == 0:  # ~7% positive samples
                category = "positive"
            else:
                # Weight other categories
                weights = {
                    "anxiety": 15,
                    "depression": 15,
                    "family_pressure": 15,
                    "academic_stress": 20,
                    "financial_stress": 10,
                    "relationship": 10,
                    "work_stress": 10
                }
                category = random.choices(
                    list(weights.keys()),
                    weights=list(weights.values())
                )[0]

            # Get random seed from category
            seed_data = random.choice(seeds[category])
            user_input, emotion, risk_level = seed_data

            # Generate sample
            print(f"\n[{i+1}/{num_samples}] Generating {category} sample...")
            sample = self.generate_sample(user_input, category, emotion, risk_level)

            if sample:
                sample["category"] = category
                dataset.append(sample)
                print(f"✓ Generated: {sample['input'][:60]}...")
            else:
                print(f"✗ Failed to generate sample")

            # Rate limiting (be nice to APIs)
            time.sleep(1)  # 1 second between requests

            # Save checkpoint
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(dataset, output_file)
                print(f"\n✓ Checkpoint: {len(dataset)} samples saved")

        # Final save
        self._save_dataset(dataset, output_file)

        print("\n" + "=" * 60)
        print(f"✓ Dataset generation complete!")
        print(f"✓ Total samples: {len(dataset)}")
        print(f"✓ Saved to: {output_file}")
        print("=" * 60)

        return dataset

    def _save_checkpoint(self, dataset: List[Dict], output_file: str):
        """Save intermediate checkpoint"""
        checkpoint_file = output_file.replace(".json", "_checkpoint.json")
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({
                "samples": dataset,
                "count": len(dataset),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2, ensure_ascii=False)

    def _save_dataset(self, dataset: List[Dict], output_file: str):
        """Save final dataset"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Calculate statistics
        categories = {}
        emotions = {}
        risk_levels = {}

        for sample in dataset:
            cat = sample.get("category", "unknown")
            emo = sample.get("emotion", "unknown")
            risk = sample.get("risk_level", "unknown")

            categories[cat] = categories.get(cat, 0) + 1
            emotions[emo] = emotions.get(emo, 0) + 1
            risk_levels[risk] = risk_levels.get(risk, 0) + 1

        # Save dataset with metadata
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_samples": len(dataset),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "provider": self.provider,
                    "model": self.model,
                    "statistics": {
                        "categories": categories,
                        "emotions": emotions,
                        "risk_levels": risk_levels
                    }
                },
                "samples": dataset
            }, f, indent=2, ensure_ascii=False)


def print_statistics(dataset: List[Dict]):
    """Print dataset statistics"""
    categories = {}
    emotions = {}
    risk_levels = {}

    for sample in dataset:
        cat = sample.get("category", "unknown")
        emo = sample.get("emotion", "unknown")
        risk = sample.get("risk_level", "unknown")

        categories[cat] = categories.get(cat, 0) + 1
        emotions[emo] = emotions.get(emo, 0) + 1
        risk_levels[risk] = risk_levels.get(risk, 0) + 1

    total = len(dataset)

    print("\n📊 Dataset Statistics:")
    print(f"\nTotal Samples: {total}")

    print(f"\nCategories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:20} {count:4} ({count/total*100:5.1f}%)")

    print(f"\nEmotions:")
    for emo, count in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"  {emo:20} {count:4} ({count/total*100:5.1f}%)")

    print(f"\nRisk Levels:")
    for risk, count in sorted(risk_levels.items(), key=lambda x: -x[1]):
        print(f"  {risk:20} {count:4} ({count/total*100:5.1f}%)")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Generate synthetic mental health dataset")
    parser.add_argument("--provider", choices=["openai", "claude", "gemini"],
                       default="claude", help="LLM provider to use")
    parser.add_argument("--api-key", help="API key (or set via environment variable)")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--output", default="../data/synthetic_training_data.json",
                       help="Output file path")
    parser.add_argument("--checkpoint", type=int, default=100,
                       help="Checkpoint interval")

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key
    if not api_key:
        env_vars = {
            "claude": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_API_KEY"]  # Try multiple names
        }

        env_var_options = env_vars.get(args.provider)

        # For gemini, try multiple environment variable names
        if args.provider == "gemini":
            for var_name in env_var_options:
                api_key = os.getenv(var_name)
                if api_key:
                    print(f"✓ Found API key in {var_name}")
                    break
        else:
            env_var = env_var_options
            api_key = os.getenv(env_var)

        if not api_key:
            print(f"Error: API key not found!")
            if args.provider == "gemini":
                print(f"Set one of these environment variables:")
                print(f"  export GEMINI_API_KEY=your-key")
                print(f"  export GOOGLE_API_KEY=your-key")
            else:
                print(f"Set environment variable: export {env_var_options}=your-key")
            print(f"Or use: --api-key your-key")
            print(f"\nTo get a FREE Gemini API key:")
            print(f"  1. Go to https://aistudio.google.com/app/apikey")
            print(f"  2. Click 'Get API Key' or 'Create API Key'")
            print(f"  3. Copy the key and set it as environment variable")
            return

    # Initialize generator
    try:
        generator = DatasetGenerator(api_key=api_key, provider=args.provider)
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return

    # Generate dataset
    dataset = generator.generate_dataset(
        num_samples=args.num_samples,
        output_file=args.output,
        checkpoint_interval=args.checkpoint
    )

    # Print statistics
    print_statistics(dataset)

    print("\n✅ Done! Next steps:")
    print("1. Review samples manually (check quality)")
    print("2. Run: python scripts/validate_dataset.py")
    print("3. Format for training: python scripts/format_dataset.py")
    print("4. Train model: python backend/train_model_lora.py")


if __name__ == "__main__":
    main()

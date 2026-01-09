"""
SafeMind AI - Synthetic Dataset Generator
Generates culturally-aware mental health training conversations

Uses Groq API (FREE, FAST, NO CREDIT CARD REQUIRED!)
- No traffic issues
- Very fast generation (2-3x faster than competitors)
- High quality outputs
- Completely FREE

QUICK START:
1. Get FREE Groq API key: https://console.groq.com/keys
2. Set environment variable:
   export GROQ_API_KEY=your-key-here
3. Run:
   python3 generate_dataset.py --num-samples 500 --output ../data/mental_health_dataset.json

USAGE:
  python3 generate_dataset.py --num-samples 500
  python3 generate_dataset.py --num-samples 1000
  python3 generate_dataset.py --api-key YOUR_KEY --num-samples 500
"""

import json
import os
import time
from typing import List, Dict, Optional
import random
import argparse
import requests

class DatasetGenerator:
    """Generate synthetic mental health conversation dataset using Groq"""

    def __init__(self, api_key: str):
        """
        Initialize dataset generator with Groq API

        Args:
            api_key: Groq API key (get free at https://console.groq.com/keys)
        """
        self.api_key = api_key
        # Using Llama 3 70B - best quality free model
        self.model = "llama3-70b-8192"
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

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
8. If high risk: Include crisis resources (1333 hotline, Sumithrayo: 011-2696666)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "instruction": "You are a mental health awareness chatbot for Sri Lankan users. Provide empathetic, culturally-aware support without diagnosing or replacing professional care.",
  "input": "[Generated user message - make it realistic and different from example]",
  "response": "[Your empathetic, culturally-appropriate response]",
  "emotion": "{emotion}",
  "risk_level": "{risk_level}"
}}"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"}
            }

            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Parse JSON from response
                content = content.replace("```json", "").replace("```", "").strip()
                sample = json.loads(content)

                # Validate required fields
                required_fields = ["instruction", "input", "response", "emotion", "risk_level"]
                if all(field in sample for field in required_fields):
                    return sample
                else:
                    print(f"Warning: Missing fields in generated sample")
                    return None
            else:
                print(f"Error: Groq API returned {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"Error generating sample: {e}")
            return None

    def generate_dataset(self, num_samples: int = 1000,
                        output_file: str = "../data/mental_health_dataset.json",
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

        print("=" * 70)
        print(f"Generating {num_samples} synthetic training samples...")
        print(f"Using Groq API | Model: {self.model}")
        print("=" * 70)

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

            # Rate limiting (be nice to APIs) - Groq is fast so minimal delay
            time.sleep(0.5)  # 0.5 second between requests

            # Save checkpoint
            if (i + 1) % checkpoint_interval == 0:
                self._save_checkpoint(dataset, output_file)
                print(f"\n✓ Checkpoint: {len(dataset)} samples saved")

        # Final save
        self._save_dataset(dataset, output_file)

        print("\n" + "=" * 70)
        print(f"✓ Dataset generation complete!")
        print(f"✓ Total samples: {len(dataset)}")
        print(f"✓ Saved to: {output_file}")
        print("=" * 70)

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
                    "provider": "groq",
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
    parser = argparse.ArgumentParser(
        description="Generate synthetic mental health dataset using Groq (FREE & FAST)"
    )
    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY env variable)")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of samples to generate (default: 1000)")
    parser.add_argument("--output", default="../data/mental_health_dataset.json",
                       help="Output file path")
    parser.add_argument("--checkpoint", type=int, default=100,
                       help="Checkpoint interval (default: 100)")

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('GROQ_API_KEY')

    if not api_key:
        print("=" * 70)
        print("ERROR: Groq API key not found!")
        print("=" * 70)
        print("\nTo get a FREE Groq API key:")
        print("  1. Go to https://console.groq.com/keys")
        print("  2. Sign up (FREE, no credit card required)")
        print("  3. Click 'Create API Key'")
        print("  4. Copy the key")
        print("\nThen set it:")
        print("  export GROQ_API_KEY=your-key-here")
        print("Or:")
        print("  python3 generate_dataset.py --api-key your-key-here")
        print("\n" + "=" * 70)
        return

    # Initialize generator
    try:
        generator = DatasetGenerator(api_key=api_key)
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
    print("1. Review samples: cat ../data/mental_health_dataset.json")
    print("2. Validate quality: Check a few samples manually")
    print("3. Train model: python3 ../backend/train_model.py")
    print("4. Fine-tune model: python3 ../backend/fine_tune_model.py")


if __name__ == "__main__":
    main()

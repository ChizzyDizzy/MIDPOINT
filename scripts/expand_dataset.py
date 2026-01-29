"""
SafeMind AI - Dataset Expander
Expands existing JSON templates into 1500-sample training dataset

NO API calls needed - uses existing data files only!

USAGE:
  python3 expand_dataset.py --output ../data/mental_health_dataset.json --num-samples 1500
"""

import json
import os
import random
import time
from typing import List, Dict
import argparse


class DatasetExpander:
    """Expand existing JSON templates into comprehensive training dataset"""

    def __init__(self, data_dir: str = "../data"):
        """Initialize by loading all template files"""
        self.data_dir = data_dir

        # Load all template files
        self.crisis_patterns = self._load_json("crisis_patterns.json")
        self.cultural_templates = self._load_json("cultural_templates.json")
        self.enhanced_crisis = self._load_json("enhanced_crisis_patterns.json")
        self.mental_health_convos = self._load_json("mental_health_conversations.json")
        self.response_templates = self._load_json("response_templates.json")
        self.training_convos = self._load_json("training_conversations.json")

        print("✓ Loaded all template files")

    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file from data directory"""
        path = os.path.join(self.data_dir, filename)
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_sri_lankan_variations(self, base_text: str, category: str) -> List[str]:
        """Generate Sri Lankan cultural variations"""
        variations = [base_text]

        # Academic stress variations (very common in SL)
        academic_additions = [
            " especially with A/L exams coming up",
            " and my parents keep comparing me to others",
            " because of the competition for university admission",
            " and I'm worried about disappointing my family",
            " with all the pressure to get into a good university"
        ]

        # Family pressure variations
        family_additions = [
            " but my family doesn't understand",
            " and my parents have high expectations",
            " because of family obligations",
            " and I feel like I'm letting everyone down",
            " but I can't talk to my family about it"
        ]

        # Cultural context additions
        cultural_additions = [
            " and people will judge me if they find out",
            " but mental health is stigmatized in our community",
            " and I don't want to bring shame to my family",
            " but everyone expects me to just be strong",
            " and I feel like I have to hide how I'm feeling"
        ]

        if category in ["anxiety", "stress", "academic_stress"]:
            for addition in random.sample(academic_additions, 2):
                variations.append(base_text + addition)

        if category in ["family_issues", "cultural_pressure", "depression"]:
            for addition in random.sample(family_additions, 2):
                variations.append(base_text + addition)

        if category in ["loneliness", "isolation"]:
            for addition in random.sample(cultural_additions, 2):
                variations.append(base_text + addition)

        return variations

    def _generate_response(self, category: str, risk_level: str, user_input: str) -> str:
        """Generate culturally appropriate response"""

        # Get base response template
        response_category = category
        if category in ["family_issues", "cultural_pressure", "academic_stress", "financial_stress"]:
            response_category = "stress"
        elif category in ["isolation", "grief"]:
            response_category = "loneliness"
        elif category in ["crisis", "self_harm"]:
            response_category = "crisis"

        base_responses = self.response_templates.get(response_category, self.response_templates["general"])
        response = random.choice(base_responses)

        # Add culturally-aware additions
        cultural = self.cultural_templates["south_asian"]

        # Add empathetic continuation
        continuations = [
            " In our culture, there's often pressure to appear strong, but it's okay to acknowledge these feelings.",
            " Many people in our community face similar challenges, and seeking support is a sign of strength.",
            " Your feelings are valid, regardless of what others might say.",
            " It takes courage to recognize when things are difficult."
        ]
        response += random.choice(continuations)

        # Add specific guidance based on risk level
        if risk_level == "low":
            coping = [
                " Have you tried talking to someone you trust about this?",
                " Sometimes taking small steps can help - what's one thing you could try today?",
                " Would you be open to exploring some coping strategies that might help?"
            ]
            response += random.choice(coping)

        elif risk_level == "medium":
            response += " " + cultural["family_support_text"]
            response += " It might also be helpful to speak with a counselor or mental health professional."

        elif risk_level == "high":
            response += "\n\n**I'm concerned about your wellbeing. Please consider reaching out to:**"
            response += f"\n- Mental Health Helpline: {cultural['resources']['helplines']['mental_health']}"
            response += f"\n- National Crisis Line: {cultural['resources']['helplines']['sri_lanka']}"
            response += "\n- A trusted family member, friend, or counselor"

        elif risk_level == "immediate":
            response = "I'm very concerned about what you're sharing. Your safety is the most important thing right now."
            response += "\n\n🚨 **IMMEDIATE HELP NEEDED:**"
            response += f"\n- **National Crisis Hotline (Sri Lanka): {cultural['resources']['helplines']['sri_lanka']}**"
            response += "\n- **Sumithrayo (24/7): 011-2696666**"
            response += "\n- **Emergency Services: 119**"
            response += "\n\nPlease reach out to one of these services immediately. You don't have to face this alone."

        return response

    def _create_training_samples(self, num_samples: int = 1500) -> List[Dict]:
        """Create training samples from templates"""

        samples = []

        # Define categories and their proportional weights (sum to 1.0)
        category_weights = {
            "anxiety": 0.167,
            "depression": 0.133,
            "stress": 0.133,
            "academic_stress": 0.133,
            "family_issues": 0.100,
            "cultural_pressure": 0.067,
            "loneliness": 0.067,
            "financial_stress": 0.053,
            "relationship": 0.053,
            "crisis": 0.033,
            "self_harm": 0.020,
            "positive": 0.040,
        }

        # Calculate actual counts from weights
        category_distribution = {}
        assigned = 0
        for cat, weight in category_weights.items():
            count = int(num_samples * weight)
            category_distribution[cat] = count
            assigned += count
        # Distribute remainder to largest category
        if assigned < num_samples:
            category_distribution["anxiety"] += (num_samples - assigned)

        # Base inputs for each category
        category_inputs = {
            "anxiety": [
                "I've been feeling really anxious lately",
                "I can't stop worrying about everything",
                "My heart races and I feel panicky",
                "I'm constantly nervous and on edge",
                "I have panic attacks frequently",
                "I worry all the time and can't relax",
                "my anxiety is through the roof",
                "can't shake this anxious feeling",
                "everything makes me anxious these days",
                "i get so nervous i feel sick",
                "my mind won't stop racing",
                "i feel like something bad is going to happen",
                "the anxiety is overwhelming me",
                "i'm scared all the time for no reason",
                "my chest gets tight when i'm stressed",
                "social situations make me so anxious",
                "i overthink everything constantly",
                "i can't control my worrying",
                "anxiety keeps me up at night",
            ],
            "depression": [
                "I feel sad all the time",
                "Nothing makes me happy anymore",
                "I have no energy to do anything",
                "I feel empty inside",
                "I've lost interest in everything I used to enjoy",
                "I feel hopeless about the future",
                "everything feels pointless",
                "i don't enjoy anything anymore",
                "i'm just so tired of everything",
                "why even bother getting out of bed",
                "life feels meaningless",
                "i feel numb all the time",
                "nothing brings me joy",
                "i've been crying a lot lately",
                "i feel like a failure",
                "the sadness won't go away",
                "i'm tired of pretending to be okay",
                "i can't see things getting better",
            ],
            "stress": [
                "I'm under so much pressure",
                "Everything feels overwhelming",
                "I can't handle all this stress",
                "I'm so stressed I can't think straight",
                "The stress is affecting my health",
                "I feel like I'm drowning in responsibilities",
                "there's too much on my plate",
                "i can't catch a break",
                "stress is eating me alive",
                "i'm at my breaking point",
                "everything is piling up",
                "the pressure is too much",
                "i'm burned out completely",
                "i have no time for myself",
                "everyone wants something from me",
                "i'm running on empty",
            ],
            "academic_stress": [
                "I'm stressed about my A/L exams",
                "I failed my university exam",
                "The pressure to get high marks is too much",
                "I can't handle the academic workload",
                "I'm afraid I'll fail and disappoint everyone",
                "University admissions are making me anxious",
                "My exam results are causing me so much stress",
                "i bombed my exam and feel terrible",
                "my grades are slipping and i can't stop it",
                "everyone expects me to get top marks",
                "i'm not smart enough for this course",
                "the competition is so intense i can't cope",
                "my future depends on these results",
                "i can't focus on studying anymore",
                "exam pressure is crushing me",
                "what if i don't get into university",
                "my parents will be so disappointed if i fail",
                "A/Ls are destroying my mental health",
            ],
            "family_issues": [
                "My parents don't understand me",
                "My family has too many expectations",
                "There's constant conflict at home",
                "My parents want me to be someone I'm not",
                "I feel trapped by family obligations",
                "My family doesn't support my choices",
                "my parents are always fighting",
                "home doesn't feel safe anymore",
                "they compare me to my siblings all the time",
                "nothing i do is good enough for them",
                "i wish my parents would listen to me",
                "they don't take my feelings seriously",
                "i feel invisible at home",
                "my parents control everything i do",
                "i can't be myself around family",
                "family pressure is suffocating me",
            ],
            "cultural_pressure": [
                "My family wants me to follow a certain career path",
                "There's pressure to get married",
                "People judge me for my choices",
                "I feel torn between my culture and what I want",
                "Society's expectations are overwhelming",
                "I can't be myself because of cultural norms",
                "everyone has opinions about my life",
                "arranged marriage pressure is too much",
                "i can't pursue my passion because of family",
                "what will people say about me",
                "our community is so judgmental",
                "i have to hide who i really am",
                "i feel like i'm living for others not myself",
                "cultural expectations are suffocating",
            ],
            "loneliness": [
                "I feel so alone",
                "I have no one to talk to",
                "I feel isolated even around people",
                "Nobody understands what I'm going through",
                "I feel disconnected from everyone",
                "I'm lonely and it hurts",
                "i don't have any real friends",
                "everyone has someone but me",
                "i feel invisible to the world",
                "no one reaches out to me",
                "i'm surrounded by people but still lonely",
                "i miss having someone who cares",
                "nobody would notice if i disappeared",
                "i don't fit in anywhere",
            ],
            "financial_stress": [
                "My family is struggling financially",
                "I can't find a job after graduation",
                "I worry about money constantly",
                "I can't afford university fees",
                "Financial problems are overwhelming me",
                "The cost of living is too much",
                "money stress is killing me",
                "i don't know how we'll pay the bills",
                "i feel guilty for being a financial burden",
                "i work so hard but it's never enough",
                "financial stress affects everything in my life",
                "debt is crushing me",
            ],
            "relationship": [
                "I'm having problems in my relationship",
                "I feel like nobody cares about me",
                "My relationships are falling apart",
                "I don't know how to fix my relationship",
                "I feel abandoned by people I trusted",
                "my partner doesn't understand me",
                "we fight all the time now",
                "i feel unloved by everyone",
                "trust issues are ruining my relationships",
                "i got my heart broken badly",
                "i can't move on from the breakup",
                "i'm scared of getting hurt again",
            ],
            "crisis": [
                "I don't see the point in living anymore",
                "I feel completely hopeless",
                "I don't think I can go on",
                "I feel like a burden to everyone",
                "Nothing will ever get better",
                "I don't want to be here anymore",
                "i want everything to end",
                "i can't do this anymore",
                "what's the point of living",
                "everyone would be better off without me",
                "i'm tired of existing",
                "life is unbearable",
            ],
            "self_harm": [
                "I've been hurting myself",
                "I can't stop thinking about self-harm",
                "I hurt myself when things get too much",
                "I don't know how to cope without hurting myself",
                "cutting helps me feel something",
                "i use pain to distract from emotions",
                "self-harm is my only release",
                "i've been hiding the scars",
            ],
            "positive": [
                "I've been feeling better lately",
                "The breathing exercises helped",
                "I talked to someone and it helped",
                "I'm making progress",
                "Things are getting a bit better",
                "Thank you for your support",
                "i tried what you suggested and it worked",
                "today was actually okay",
                "i'm starting to feel hopeful again",
                "small steps are helping me",
                "therapy is really helping",
                "i feel a little lighter today",
            ]
        }

        # Emotion mappings
        category_emotions = {
            "anxiety": ["anxious", "nervous", "worried", "panicked"],
            "depression": ["sad", "hopeless", "empty", "numb"],
            "stress": ["stressed", "overwhelmed", "pressured", "exhausted"],
            "academic_stress": ["anxious", "stressed", "worried", "pressured"],
            "family_issues": ["frustrated", "sad", "trapped", "misunderstood"],
            "cultural_pressure": ["conflicted", "pressured", "stressed", "trapped"],
            "loneliness": ["lonely", "isolated", "abandoned", "disconnected"],
            "financial_stress": ["worried", "stressed", "anxious", "desperate"],
            "relationship": ["hurt", "sad", "abandoned", "lonely"],
            "crisis": ["hopeless", "despair", "suicidal", "broken"],
            "self_harm": ["desperate", "overwhelmed", "hopeless", "numb"],
            "positive": ["hopeful", "relieved", "grateful", "better"]
        }

        # Risk level distribution per category
        category_risk_levels = {
            "anxiety": ["low", "low", "low", "medium", "medium"],
            "depression": ["medium", "medium", "medium", "high", "high"],
            "stress": ["low", "low", "medium", "medium"],
            "academic_stress": ["low", "low", "medium", "medium", "medium"],
            "family_issues": ["low", "low", "medium", "medium"],
            "cultural_pressure": ["low", "medium", "medium"],
            "loneliness": ["medium", "medium", "high"],
            "financial_stress": ["medium", "medium", "high"],
            "relationship": ["low", "medium", "medium"],
            "crisis": ["high", "immediate"],
            "self_harm": ["high", "immediate"],
            "positive": ["low"]
        }

        print("=" * 70)
        print(f"Expanding templates into {num_samples} training samples...")
        print("=" * 70)

        sample_id = 0
        for category, count in category_distribution.items():
            if len(samples) >= num_samples:
                break

            base_inputs = category_inputs.get(category, ["I need help"])

            for i in range(count):
                if len(samples) >= num_samples:
                    break

                # Select base input
                base_input = random.choice(base_inputs)

                # Generate variations
                variations = self._get_sri_lankan_variations(base_input, category)
                user_input = random.choice(variations)

                # Determine risk level
                risk_levels = category_risk_levels.get(category, ["low", "medium"])
                risk_level = random.choice(risk_levels)

                # Select emotion
                emotions = category_emotions.get(category, ["neutral"])
                emotion = random.choice(emotions)

                # Generate response
                response = self._generate_response(category, risk_level, user_input)

                # Create sample
                sample = {
                    "instruction": "You are SafeMind AI, a mental health awareness chatbot for Sri Lankan users. Provide empathetic, culturally-aware support without diagnosing or replacing professional care. Be sensitive to family dynamics, academic pressure, and social stigma around mental health in Sri Lankan culture.",
                    "input": user_input,
                    "response": response,
                    "emotion": emotion,
                    "risk_level": risk_level,
                    "category": category
                }

                samples.append(sample)
                sample_id += 1

                if sample_id % 100 == 0:
                    print(f"[{sample_id}/{num_samples}] Generated {category} sample")

        # Shuffle to mix categories
        random.shuffle(samples)

        print(f"\n✓ Generated {len(samples)} training samples")
        return samples[:num_samples]

    def generate_dataset(self, num_samples: int = 1500,
                        output_file: str = "../data/mental_health_dataset.json") -> List[Dict]:
        """Generate complete dataset"""

        # Generate samples
        samples = self._create_training_samples(num_samples)

        # Calculate statistics
        stats = self._calculate_statistics(samples)

        # Save dataset
        self._save_dataset(samples, stats, output_file)

        print("\n" + "=" * 70)
        print(f"✓ Dataset expansion complete!")
        print(f"✓ Total samples: {len(samples)}")
        print(f"✓ Saved to: {output_file}")
        print("=" * 70)

        return samples

    def _calculate_statistics(self, samples: List[Dict]) -> Dict:
        """Calculate dataset statistics"""
        categories = {}
        emotions = {}
        risk_levels = {}

        for sample in samples:
            cat = sample.get("category", "unknown")
            emo = sample.get("emotion", "unknown")
            risk = sample.get("risk_level", "unknown")

            categories[cat] = categories.get(cat, 0) + 1
            emotions[emo] = emotions.get(emo, 0) + 1
            risk_levels[risk] = risk_levels.get(risk, 0) + 1

        return {
            "categories": categories,
            "emotions": emotions,
            "risk_levels": risk_levels
        }

    def _save_dataset(self, samples: List[Dict], stats: Dict, output_file: str):
        """Save dataset to file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_samples": len(samples),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": "template_expansion",
                    "source": "existing_json_files",
                    "statistics": stats
                },
                "samples": samples
            }, f, indent=2, ensure_ascii=False)


def print_statistics(samples: List[Dict]):
    """Print dataset statistics"""
    categories = {}
    emotions = {}
    risk_levels = {}

    for sample in samples:
        cat = sample.get("category", "unknown")
        emo = sample.get("emotion", "unknown")
        risk = sample.get("risk_level", "unknown")

        categories[cat] = categories.get(cat, 0) + 1
        emotions[emo] = emotions.get(emo, 0) + 1
        risk_levels[risk] = risk_levels.get(risk, 0) + 1

    total = len(samples)

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
        description="Expand existing JSON templates into training dataset (NO API needed!)"
    )
    parser.add_argument("--num-samples", type=int, default=1500,
                       help="Number of samples to generate (default: 1500)")
    parser.add_argument("--output", default="../data/mental_health_dataset.json",
                       help="Output file path")
    parser.add_argument("--data-dir", default="../data",
                       help="Directory containing template JSON files")

    args = parser.parse_args()

    # Initialize expander
    expander = DatasetExpander(data_dir=args.data_dir)

    # Generate dataset
    samples = expander.generate_dataset(
        num_samples=args.num_samples,
        output_file=args.output
    )

    # Print statistics
    print_statistics(samples)

    print("\n✅ Done! Next steps:")
    print("1. Review samples:")
    print(f"   python3 -c \"import json; data=json.load(open('{args.output}')); print(json.dumps(data['samples'][0], indent=2))\"")
    print("2. Train model:")
    print("   cd ../backend && python3 train_model.py")
    print("3. Fine-tune model:")
    print("   cd ../backend && python3 fine_tune_lora.py")


if __name__ == "__main__":
    main()

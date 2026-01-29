"""
SafeMind AI - Model Fine-Tuning Script
Train a custom mental health chatbot using your synthetic dataset

This script fine-tunes a conversational AI model on your mental health dataset
to create a model specifically trained for SafeMind AI responses.
"""

import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

print("=" * 70)
print(" " * 15 + "SafeMind AI - Model Training")
print("=" * 70)
print()

# Configuration
BASE_MODEL = "microsoft/DialoGPT-small"  # Can change to DialoGPT-medium for better quality
OUTPUT_DIR = "./safemind-mental-health-model"
DATASET_PATH = "../data/training_conversations.json"
NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 5e-5

# Step 1: Load your synthetic dataset
print("[1/7] Loading synthetic dataset...")
print("-" * 70)

try:
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"❌ ERROR: Dataset not found at {DATASET_PATH}")
    print("Please ensure training_conversations.json exists in the data/ folder")
    exit(1)

conversations = data['conversations']
print(f"✓ Loaded {len(conversations)} training conversations")
print(f"  Categories: {set(c['category'] for c in conversations)}")
print()

# Step 2: Prepare training data
print("[2/7] Preparing training data...")
print("-" * 70)

# Create empathetic response templates based on mental health best practices
response_templates = {
    'anxiety': [
        "I hear that you're feeling anxious. That sounds really challenging. Anxiety can feel overwhelming, but you're not alone in this. Would you like to talk more about what's triggering these feelings?",
        "Thank you for sharing that with me. Anxiety about {topic} is completely valid. Many people experience similar feelings. What helps you feel calmer when these anxious thoughts come up?",
        "I understand you're going through a difficult time with anxiety. It takes courage to talk about these feelings. Let's explore what might help you manage this together."
    ],
    'depression': [
        "Thank you for opening up about this. Depression can make everything feel heavy and difficult. I'm here to listen and support you. How long have you been feeling this way?",
        "I hear you, and your feelings are completely valid. Depression affects many people, and it's okay to not be okay. Have you been able to talk to anyone else about how you're feeling?",
        "It sounds like you're going through a really tough time. I'm glad you're reaching out. Small steps can make a difference - what's one thing that used to bring you even a little bit of joy?"
    ],
    'crisis': [
        "I'm really concerned about what you're sharing with me. Your safety is the most important thing right now. I want to make sure you get the immediate support you need. Are you in a safe place right now?",
        "Thank you for trusting me with this. What you're going through sounds incredibly painful, and I want to help connect you with someone who can provide immediate support. Can we talk about reaching out to a crisis counselor?",
        "I hear how much pain you're in, and I'm taking this very seriously. You don't have to go through this alone. There are people available 24/7 who want to help. Would you be willing to call a crisis helpline with me?"
    ],
    'stress': [
        "It sounds like you're dealing with a lot of stress right now. Feeling overwhelmed is completely understandable when there's so much on your plate. What's weighing on you the most?",
        "I hear you - stress can feel all-consuming sometimes. Let's break this down together. What are the main sources of stress you're facing?",
        "That's a lot to handle. Stress can affect us physically and emotionally. Have you found anything that helps you feel less overwhelmed, even temporarily?"
    ],
    'loneliness': [
        "Feeling lonely can be really painful. I'm here with you, and I want you to know that your feelings matter. Tell me more about what's making you feel isolated?",
        "Thank you for sharing this with me. Loneliness is something many people struggle with, especially in our fast-paced world. What kinds of connections are you missing most?",
        "I hear that you're feeling alone. That's such a difficult feeling to carry. Let's talk about your social connections - what would meaningful connection look like for you?"
    ],
    'cultural_pressure': [
        "I understand that family expectations and cultural pressures can feel overwhelming. It's challenging when you feel caught between different expectations. How is this affecting you?",
        "Thank you for sharing this. Cultural and family expectations can create unique pressures. Your feelings about this are valid, even if they conflict with what's expected. What would you like to talk about?",
        "I hear you navigating between cultural expectations and your own needs. That balance can be really difficult. You're not alone in feeling this way."
    ],
    'general': [
        "Thank you for sharing that with me. I'm here to listen and support you. Tell me more about what's on your mind.",
        "I hear you. It takes courage to talk about what you're experiencing. I'm here for you. What would be most helpful to talk about right now?",
        "I'm listening. Your feelings and experiences matter. What's been weighing on you?"
    ]
}

training_texts = []
for conv in conversations:
    user_input = conv['user_input']
    category = conv.get('category', 'general')

    # Select appropriate response template
    if category in response_templates:
        import random
        response_template = random.choice(response_templates[category])
        # Simple substitution if template has {topic}
        response = response_template.replace('{topic}', user_input.lower())
    else:
        response = f"I hear you. Thank you for sharing that with me. Tell me more about {user_input.lower()}."

    # Create training format for conversational model
    # Format: User: <message>\nAssistant: <response>\n
    training_text = f"User: {user_input}\nAssistant: {response}\n"
    training_texts.append(training_text)

print(f"✓ Prepared {len(training_texts)} training examples")
print(f"  Average length: {sum(len(t) for t in training_texts) // len(training_texts)} characters")
print()

# Step 3: Load base model
print("[3/7] Loading base model...")
print("-" * 70)
print(f"Model: {BASE_MODEL}")
print("This may take a few minutes on first run (downloading ~500MB)...")
print()

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # Set padding token (required for training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print("✓ Base model loaded successfully")
    print(f"  Model parameters: {model.num_parameters():,}")
    print()

except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure you have internet connection (first download)")
    print("2. Install dependencies: pip install transformers torch")
    print("3. Try using Google Colab for FREE GPU: https://colab.research.google.com/")
    exit(1)

# Step 4: Tokenize dataset
print("[4/7] Tokenizing dataset...")
print("-" * 70)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )

# Create Hugging Face dataset
train_dataset = Dataset.from_dict({'text': training_texts})
tokenized_dataset = train_dataset.map(
    tokenize_function,
    remove_columns=['text'],
    batched=True,
    desc="Tokenizing"
)

print(f"✓ Tokenized {len(tokenized_dataset)} examples")
print(f"  Max sequence length: 256 tokens")
print()

# Step 5: Set up training configuration
print("[5/7] Setting up training configuration...")
print("-" * 70)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training device: {device.upper()}")
if device == "cpu":
    print("⚠️  Training on CPU will be slower (~30-60 minutes)")
    print("   Consider using Google Colab for FREE GPU access")
else:
    print("✓ GPU detected - training will be faster (~10-20 minutes)")
print()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=50,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=LEARNING_RATE,
    warmup_steps=50,
    logging_steps=10,
    logging_dir='./logs',
    report_to="none",  # Disable wandb
    no_cuda=(device == "cpu")
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling, not masked
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("✓ Training configuration ready")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print()

# Step 6: Train the model
print("[6/7] Training model...")
print("=" * 70)
print()
print("📊 Training Progress:")
print("-" * 70)
print("Watch the 'loss' value - it should decrease over time.")
print("Lower loss = better model performance")
print()
print("Estimated time:")
print(f"  - GPU: ~10-20 minutes")
print(f"  - CPU: ~30-60 minutes")
print()
print("=" * 70)
print()

try:
    trainer.train()
    print()
    print("=" * 70)
    print("✓ Training completed successfully!")
    print("=" * 70)
    print()

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    print("Partial progress has been saved")
    print()
except Exception as e:
    print(f"\n❌ ERROR during training: {e}")
    print("\nTroubleshooting:")
    print("1. Out of memory? Reduce BATCH_SIZE to 1")
    print("2. Use Google Colab for FREE GPU")
    print("3. Check logs in ./logs directory")
    exit(1)

# Step 7: Save the model
print("[7/7] Saving fine-tuned model...")
print("-" * 70)

try:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training info
    training_info = {
        "base_model": BASE_MODEL,
        "training_samples": len(training_texts),
        "epochs": NUM_EPOCHS,
        "categories": list(set(c['category'] for c in conversations)),
        "model_path": OUTPUT_DIR
    }

    with open(f"{OUTPUT_DIR}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)

    print(f"✓ Model saved to: {OUTPUT_DIR}")
    print(f"  - Model files: pytorch_model.bin, config.json")
    print(f"  - Tokenizer files: tokenizer.json, vocab files")
    print(f"  - Training info: training_info.json")
    print()

except Exception as e:
    print(f"❌ ERROR saving model: {e}")
    exit(1)

# Success summary
print()
print("=" * 70)
print(" " * 20 + "🎉 SUCCESS! 🎉")
print("=" * 70)
print()
print("Your SafeMind AI model has been successfully trained!")
print()
print("📊 Training Summary:")
print(f"  - Base model: {BASE_MODEL}")
print(f"  - Training samples: {len(training_texts)}")
print(f"  - Epochs completed: {NUM_EPOCHS}")
print(f"  - Model saved to: {OUTPUT_DIR}")
print()
print("🚀 Next Steps:")
print()
print("1. Update your .env file:")
print("   " + "─" * 50)
print("   AI_BACKEND=local")
print(f"   LOCAL_MODEL={OUTPUT_DIR}")
print("   " + "─" * 50)
print()
print("2. Update app_improved.py (line 10):")
print("   " + "─" * 50)
print("   from ai_model_free import SafeMindAI")
print("   " + "─" * 50)
print()
print("3. Test your model:")
print("   " + "─" * 50)
print("   python app_improved.py")
print("   python test_mvp.py")
print("   " + "─" * 50)
print()
print("=" * 70)
print()

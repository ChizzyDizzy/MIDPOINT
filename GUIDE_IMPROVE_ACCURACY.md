# SafeMind AI - Improving Accuracy Guide

This guide explains how to improve both **safety detection accuracy** and **AI response quality** in SafeMind AI.

---

## Understanding Accuracy in SafeMind AI

SafeMind AI has two accuracy components:

| Component | What it does | How to improve |
|-----------|-------------|----------------|
| **Safety Detection** | Classifies messages by risk level (none → immediate) | Add keywords, tune thresholds |
| **AI Response Quality** | Generates empathetic responses to user messages | Train with more data, fine-tune model |

---

## Part 1: Safety Detection Accuracy

The safety detector uses an **11-layer detection system** that analyzes each message through multiple checks:

1. **Immediate risk keywords** - "kill myself", "suicide", "end my life"
2. **High risk keywords** - "hurt myself", "hopeless", "worthless"
3. **Medium risk keywords** - "depressed", "panic attacks", "can't sleep"
4. **Low risk keywords** - "anxious", "stressed", "worried", "sad"
5. **Pattern matching** - Regex patterns like "want.*to.*die"
6. **Sentiment analysis** - TextBlob polarity scoring
7. **Contextual indicators** - Hopelessness, isolation, burden, pain, finality
8. **Temporal urgency** - "tonight", "now", "today"
9. **Planning indicators** - "plan", "method", "decided"
10. **Means access** - "pills", "rope", "bridge"
11. **Cultural pressure** - "parents expect", "family duty", "career pressure"

### How Risk Levels Are Determined

Each layer produces a weighted score. The highest weighted score determines the risk level:

| Weighted Score | Risk Level |
|---------------|------------|
| >= 0.9 | `immediate` |
| >= 0.7 | `high` |
| >= 0.45 | `medium` |
| >= 0.2 | `low` |
| < 0.2 | `minimal` |
| No matches | `none` |

### Adding New Keywords

To improve detection accuracy, add keywords to `data/enhanced_crisis_patterns.json`:

```json
{
  "keywords": {
    "immediate": ["suicide", "kill myself", ...],
    "high": ["hurt myself", "hopeless", ...],
    "medium": ["depressed", "panic attacks", ...],
    "low": ["anxious", "stressed", "worried", ...]
  }
}
```

**Tips for adding keywords:**
- Add keywords in **lowercase** (the detector converts all input to lowercase)
- Use **exact phrases** for multi-word entries (e.g., "hurt myself")
- Single words like "sad" will match anywhere in the message
- Test your additions using the test suite (see Part 3)

### Adding Contextual Indicators

Add new contextual patterns to detect nuanced distress signals:

```json
{
  "contextual_indicators": {
    "isolation": ["alone", "no one", "nobody", ...],
    "hopelessness": ["hopeless", "pointless", ...],
    "burden": ["burden", "everyone hates me", ...],
    "YOUR_NEW_CATEGORY": ["keyword1", "keyword2", ...]
  }
}
```

### Adding Cultural Patterns

Add Sri Lankan or South Asian specific patterns:

```json
{
  "cultural_considerations": {
    "south_asian": {
      "family_pressure": ["parents expect", "family duty", ...],
      "stigma_phrases": ["shame to family", "what will people say", ...],
      "YOUR_NEW_CATEGORY": ["keyword1", "keyword2", ...]
    }
  }
}
```

---

## Part 2: AI Response Quality

Response quality depends on whether the AI model is loaded:

| State | What happens |
|-------|-------------|
| **Model loaded** | Generates responses using fine-tuned DialoGPT |
| **Model not loaded** | Uses template-based responses from `data/response_templates.json` |

### Improving Template Responses

Edit `data/response_templates.json` to add better fallback responses:

```json
{
  "anxiety": [
    "Your response here...",
    "Another response..."
  ],
  "depression": [...],
  "stress": [...],
  "general": [...]
}
```

**Categories available:** anxiety, depression, anger, loneliness, stress, positive, general, crisis, academic_stress, family_issues, cultural_pressure, self_harm

**Tips for writing good responses:**
- Start with empathy ("I hear you", "That sounds difficult")
- Validate feelings ("Your feelings are valid")
- Ask open-ended questions ("What's been weighing on you most?")
- Avoid clinical language
- Keep responses 1-3 sentences
- Never diagnose or prescribe treatment

### Training a Better Model

The AI model is fine-tuned from DialoGPT. To improve response quality:

1. **Generate more training data** (6000+ samples):
   ```bash
   cd scripts
   python expand_dataset.py --num-samples 6000 --output ../data/mental_health_dataset.json
   ```

2. **Train on Google Colab** (see [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md)):
   - Use `DialoGPT-medium` or `DialoGPT-large` for better quality
   - Train for 5-7 epochs
   - Use learning rate 1e-5 to 2e-5
   - Watch the training loss - it should decrease each epoch

3. **Improve training data diversity** by editing `data/response_templates.json`:
   - Add more response variations per category
   - Add more natural, conversational responses
   - Include Sri Lankan cultural context where appropriate

---

## Part 3: Running the Test Suite

### Run the Full Test Suite

```bash
cd backend
python test_mvp.py
```

Select option `1` to run all 8 test cases. The test suite checks:

| Test | Input | Expected Risk |
|------|-------|--------------|
| 1 | "I feel anxious about my exam tomorrow" | `low` |
| 2 | "I feel sad all the time and nothing makes me happy anymore" | `medium` |
| 3 | "I feel hopeless and like nothing will ever get better" | `high` |
| 4 | "I want to end my life, I cant go on like this" | `immediate` |
| 5 | "I tried the breathing exercises and they really helped!" | `none` |
| 6 | "I hurt myself when things get too much" | `high` |
| 7 | "Work is really stressful and overwhelming" | `low` |
| 8 | "My family expects me to become a doctor but I want to be an artist" | `low` |

### Run a Custom Test

Select option `3` to test any message with any expected risk level.

### Quick API Test

```bash
curl -X POST http://localhost:5000/api/test \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"I feel anxious about my exam\"}"
```

---

## Part 4: Accuracy Improvement Checklist

Use this checklist to systematically improve accuracy:

- [ ] Run the test suite and note failing tests
- [ ] For each failure, identify which keywords are missing
- [ ] Add missing keywords to `data/enhanced_crisis_patterns.json`
- [ ] Add new response templates to `data/response_templates.json`
- [ ] Re-run the test suite to confirm improvements
- [ ] Generate a larger dataset (6000+ samples)
- [ ] Train the model on Google Colab (see [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md))
- [ ] Download and install the trained model
- [ ] Re-run the test suite with the trained model

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/enhanced_safety_detector.py` | 11-layer safety detection logic |
| `data/enhanced_crisis_patterns.json` | Keywords, patterns, and indicators |
| `data/response_templates.json` | Fallback response templates |
| `backend/ai_model_free.py` | AI response generation |
| `backend/test_mvp.py` | Test suite |
| `scripts/expand_dataset.py` | Training data generator |

---

## Next Steps

- **Train the model**: See [Cloud Training Guide](GUIDE_CLOUD_TRAINING.md)
- **Evaluate the model**: See [Evaluation Guide](GUIDE_EVALUATION.md)
- **Run the app**: See [macOS Guide](GUIDE_MAC.md) or [Windows Guide](GUIDE_WINDOWS.md)

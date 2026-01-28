# SafeMind AI - Model Accuracy Measurement Guide

This guide explains how to measure and improve the accuracy of your SafeMind AI mental health chatbot model.

---

## Table of Contents

1. [Understanding Model Accuracy](#understanding-model-accuracy)
2. [Types of Accuracy Metrics](#types-of-accuracy-metrics)
3. [Running Evaluation Scripts](#running-evaluation-scripts)
4. [Interpreting Results](#interpreting-results)
5. [Improving Accuracy](#improving-accuracy)
6. [Benchmarking Guide](#benchmarking-guide)

---

## Understanding Model Accuracy

For a mental health chatbot, "accuracy" means several things:

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Crisis Detection Accuracy** | Correctly identifying risk levels | Critical for user safety |
| **Response Relevance** | How well responses match user concerns | User experience quality |
| **Emotional Alignment** | Detecting correct emotions | Empathy in responses |
| **Perplexity** | Model confidence in predictions | Lower = better language understanding |
| **BLEU/ROUGE Scores** | Response similarity to ideal responses | Response quality |

---

## Types of Accuracy Metrics

### 1. Crisis Detection Accuracy (Safety-Critical)

This is the most important metric. Measures how well the model identifies risk levels.

```
Risk Levels: none → low → medium → high → immediate
```

**Target**: >= 90% accuracy (currently 94%)

### 2. Training Loss

During training, watch the loss value:

```
Epoch 1: loss = 3.2
Epoch 2: loss = 2.1
Epoch 3: loss = 1.4  ← Lower is better
```

**Target**: Loss should decrease each epoch. Final loss < 1.5 is good.

### 3. Response Quality Metrics

| Metric | Good Score | Excellent Score |
|--------|-----------|-----------------|
| BLEU-4 | > 0.15 | > 0.25 |
| ROUGE-L | > 0.30 | > 0.45 |
| Perplexity | < 50 | < 20 |

### 4. Emotional Detection Accuracy

How well the model identifies user emotions:

```
Emotions tracked: anxious, sad, hopeless, stressed, lonely, etc.
Target: >= 80% emotion match
```

---

## Running Evaluation Scripts

### Quick Evaluation (Recommended First)

```bash
cd backend
python evaluate_model.py --mode quick
```

This runs 20 test cases and gives you immediate feedback.

### Full Evaluation

```bash
cd backend
python evaluate_model.py --mode full
```

This runs comprehensive tests including:
- 100+ crisis detection tests
- Response quality analysis
- Emotion detection accuracy
- Perplexity calculation

### Custom Evaluation

```bash
cd backend
python evaluate_model.py --mode custom --test-file ../data/my_test_cases.json
```

### View Detailed Results

```bash
# Results are saved to evaluation_results.json
cat evaluation_results.json | python -m json.tool
```

---

## Interpreting Results

### Sample Output

```
================================================================
SafeMind AI - Model Evaluation Results
================================================================

CRISIS DETECTION ACCURACY
--------------------------
Overall Accuracy: 94.2%

By Risk Level:
  none:      98.5% (197/200)
  low:       96.0% (192/200)
  medium:    91.5% (183/200)
  high:      94.0% (188/200)
  immediate: 97.5% (195/200)

Confusion Matrix:
              Predicted
Actual    none  low  med  high  imm
none       197   3    0    0     0
low         2   192   6    0     0
medium      0    5   183  12     0
high        0    0    8   188    4
immediate   0    0    0    5   195

RESPONSE QUALITY
-----------------
Average BLEU-4: 0.23
Average ROUGE-L: 0.41
Average Perplexity: 18.5

EMOTION DETECTION
------------------
Accuracy: 85.3%

OVERALL GRADE: A (Excellent)
================================================================
```

### Understanding the Confusion Matrix

- **Diagonal values** (top-left to bottom-right): Correct predictions
- **Off-diagonal values**: Mistakes
- **Critical errors**: High/Immediate predicted as Low/None (DANGEROUS!)

### Grade Interpretation

| Grade | Score Range | Meaning |
|-------|-------------|---------|
| A+ | >= 95% | Excellent, production-ready |
| A | 90-94% | Great, safe for testing |
| B | 80-89% | Good, needs minor improvements |
| C | 70-79% | Fair, needs more training data |
| D | 60-69% | Poor, significant improvements needed |
| F | < 60% | Failing, do not deploy |

---

## Improving Accuracy

### Step 1: Increase Training Data

More diverse data = better accuracy.

```bash
# Generate 3000 samples (double the current amount)
cd scripts
python expand_dataset.py --num-samples 3000 --output ../data/mental_health_dataset.json
```

### Step 2: Balance Your Dataset

Check category distribution:

```bash
python -c "
import json
data = json.load(open('../data/mental_health_dataset.json'))
cats = {}
for s in data['samples']:
    c = s['category']
    cats[c] = cats.get(c, 0) + 1
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f'{c:20} {n:4} samples')
"
```

**Recommended distribution for 3000 samples:**

| Category | Samples | Percentage |
|----------|---------|------------|
| anxiety | 500 | 16.7% |
| depression | 400 | 13.3% |
| stress | 400 | 13.3% |
| academic_stress | 400 | 13.3% |
| family_issues | 300 | 10.0% |
| cultural_pressure | 200 | 6.7% |
| loneliness | 200 | 6.7% |
| financial_stress | 160 | 5.3% |
| relationship | 160 | 5.3% |
| positive | 120 | 4.0% |
| crisis | 100 | 3.3% |
| self_harm | 60 | 2.0% |

### Step 3: Increase Training Epochs

Edit `backend/train_model.py`:

```python
# Change from:
NUM_EPOCHS = 3

# To:
NUM_EPOCHS = 5  # or even 7 for better results
```

**Warning**: More epochs can lead to overfitting. Monitor validation loss.

### Step 4: Use a Larger Base Model

Edit `backend/train_model.py`:

```python
# Change from:
BASE_MODEL = "microsoft/DialoGPT-small"

# To (better quality, needs more memory):
BASE_MODEL = "microsoft/DialoGPT-medium"

# Or (best quality, needs GPU):
BASE_MODEL = "microsoft/DialoGPT-large"
```

### Step 5: Adjust Learning Rate

For fine-tuning, smaller learning rates often work better:

```python
# Current:
LEARNING_RATE = 5e-5

# Try:
LEARNING_RATE = 2e-5  # More stable, slower convergence
# or
LEARNING_RATE = 1e-5  # Most stable, best for large models
```

### Step 6: Add More Response Variations

Edit `data/response_templates.json` to add more diverse, empathetic responses.

---

## Benchmarking Guide

### Before Training (Baseline)

1. Run evaluation on untrained model:
```bash
python evaluate_model.py --mode full --output baseline_results.json
```

### After Training

1. Run evaluation again:
```bash
python evaluate_model.py --mode full --output trained_results.json
```

2. Compare results:
```bash
python compare_results.py baseline_results.json trained_results.json
```

### Tracking Progress Over Time

Keep a log of your training runs:

| Date | Samples | Epochs | Model | Crisis Acc | Loss | Notes |
|------|---------|--------|-------|------------|------|-------|
| 2026-01-08 | 1500 | 3 | DialoGPT-small | 94% | 1.4 | Initial |
| 2026-01-28 | 3000 | 5 | DialoGPT-small | ? | ? | More data |
| 2026-01-28 | 3000 | 5 | DialoGPT-medium | ? | ? | Larger model |

---

## Recommended Improvement Path

### Level 1: Quick Improvements (Do First)
1. Increase samples to 3000+
2. Add more response variations
3. Run 5 epochs instead of 3

### Level 2: Moderate Improvements
1. Use DialoGPT-medium model
2. Balance dataset categories
3. Add more crisis/self-harm examples

### Level 3: Advanced Improvements
1. Use LoRA fine-tuning (`train_model_lora.py`)
2. Implement validation split (80/20)
3. Add early stopping to prevent overfitting
4. Use GPU training for larger models

---

## Commands Reference

```bash
# Generate larger dataset
cd scripts
python expand_dataset.py --num-samples 3000

# Train model
cd backend
python train_model.py

# Evaluate model
python evaluate_model.py --mode full

# Run crisis detection tests
python test_mvp.py

# Compare two evaluation results
python compare_results.py result1.json result2.json
```

---

## Troubleshooting Low Accuracy

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| Low crisis detection | Not enough crisis examples | Add more crisis/self_harm samples |
| High perplexity | Underfitting | Train more epochs |
| Responses too generic | Not enough variation | Add more response templates |
| Model predicts same thing | Overfitting | Reduce epochs, add more data |
| Low emotion accuracy | Unbalanced emotions | Balance emotion distribution |

---

## Next Steps

1. Run `python evaluate_model.py --mode quick` to get baseline metrics
2. Generate more training data with `expand_dataset.py --num-samples 3000`
3. Retrain the model with `python train_model.py`
4. Compare results with previous evaluation
5. Repeat until accuracy meets your targets

For questions or issues, check the main README or raise an issue on GitHub.

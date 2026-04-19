# cot-reasoning-evaluation

# Chain-of-Thought Reasoning Evaluation

A research project done by Freysell Perez Rugama for CSE 188: Natural Language Processing at UC Merced.

This project evaluates how different prompting strategies affect the reasoning performance of Large Language Models (LLMs), with a focus on Chain-of-Thought (CoT) and Self-Consistency.

---

## Overview

Large Language Models often struggle with multi-step reasoning when using standard prompting. This project investigates:

* Does Chain-of-Thought prompting improve reasoning accuracy?
* Does self-consistency sampling further improve reliability?
* What types of errors do models make under different prompting strategies?

I compare three approaches:

| Method                   | Description                             |
| ------------------------ | --------------------------------------- |
| **standard**             | Direct answer with no reasoning         |
| **cot**                  | Step-by-step reasoning before answering |
| **cot_self_consistency** | Multiple CoT outputs + majority voting  |

---

## How It Works

The experiment:

1. Loads a set of reasoning questions (arithmetic, symbolic, multi-step)
2. Sends each question to the LLM using different prompting strategies
3. Extracts and evaluates the final answers
4. Tracks:
accuracy
error types
reasoning behavior
Saves results and generates plots automatically

---
## Project Structure

```
.
├── cot_experiment.py # Main experiment script 
├── requirements.txt  # Dependencies
├── README.md         # Project description
└── results/
    ├── initial_demo/  # First test run
    └── my_trials/
        ├── trial_1/
        ├── trial_2/
        ├── trial_3/
```
---

---

```

Each trial folder contains:

summary_results.csv        # Accuracy + error metrics
question_results.csv      # Per-question outputs
raw_outputs.jsonl         # Raw model responses
summary_report.txt        # Human-readable report
figure_accuracy_by_method.png
figure_error_types_by_method.png

```

---

## Running Experiments

1. Install dependencies

pip install -r requirements.txt

2. Install and start Ollama

ollama serve

3. Pull a model (example)

ollama pull mistral

4. Run the experiment

#### Quick demo (5 questions):

python cot_experiment.py --demo --model mistral

#### Full run:

python cot_experiment.py --model mistral

#### Run multiple trials:

python cot_experiment.py --model mistral --output results/my_trials/trial_1

python cot_experiment.py --model mistral --output results/my_trials/trial_2

python cot_experiment.py --model mistral --output results/my_trials/trial_3

### Example Results

Across multiple trials, we observe:

* Standard prompting performs poorly on multi-step reasoning
* Chain-of-Thought significantly improves accuracy
* Self-consistency improves robustness, but not always beyond CoT




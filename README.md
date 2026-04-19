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


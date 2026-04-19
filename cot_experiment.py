import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import requests


"""
main.py style single-file CoT experiment runner for an NLP final project.

What this script compares:
  1. standard               direct answer
  2. cot                    chain-of-thought prompting
  3. cot_self_consistency   sample multiple CoT responses and majority vote

Example usage:
  python cot_experiment.py
  python cot_experiment.py --demo
  python cot_experiment.py --model mistral --api-base http://localhost:11434/v1
  python cot_experiment.py --samples 7 --output results

Environment variable fallback:
  OPENAI_API_KEY
  OPENAI_API_BASE
  MODEL_NAME

Outputs:
  - raw_outputs.jsonl
  - question_results.csv
  - summary_results.csv
  - summary_report.txt
  - figure_accuracy_by_method.png
  - figure_error_types_by_method.png

This uses an OpenAI-compatible /chat/completions endpoint, so it can work with:
  - Ollama OpenAI-compatible server
  - local compatible servers
  - hosted providers with compatible APIs
"""


# -------------------------------------------------------------------
# Default question set
# -------------------------------------------------------------------

DEFAULT_QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "q1",
        "category": "arithmetic",
        "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
        "answer": "11",
    },
    {
        "id": "q2",
        "category": "arithmetic",
        "question": "A cafeteria had 23 apples. It used 20 to make lunch and then bought 6 more. How many apples does it have now?",
        "answer": "9",
    },
    {
        "id": "q3",
        "category": "arithmetic",
        "question": "A store had 64 puppies. It sold 28 and placed the rest into cages with 4 puppies in each cage. How many cages were needed?",
        "answer": "9",
    },
    {
        "id": "q4",
        "category": "rate_reasoning",
        "question": "If a person walks 2 miles in 30 minutes, how many miles do they walk in 90 minutes at the same speed?",
        "answer": "6",
    },
    {
        "id": "q5",
        "category": "counting",
        "question": "Tom has 3 red marbles and 4 blue marbles. He gives 2 blue marbles away. How many marbles does he have left in total?",
        "answer": "5",
    },
    {
        "id": "q6",
        "category": "symbolic",
        "question": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer yes or no.",
        "answer": "No",
    },
    {
        "id": "q7",
        "category": "symbolic",
        "question": "If every student in the class submitted the assignment and Maya is a student in the class, did Maya submit the assignment? Answer yes or no.",
        "answer": "Yes",
    },
    {
        "id": "q8",
        "category": "multi_step",
        "question": "A train travels 60 miles in 2 hours. At the same speed, how far will it travel in 5 hours?",
        "answer": "150",
    },
    {
        "id": "q9",
        "category": "difference",
        "question": "There are 15 trees in a grove. Workers plant trees until there are 21 trees. How many trees did they plant?",
        "answer": "6",
    },
    {
        "id": "q10",
        "category": "division",
        "question": "A ribbon is 48 inches long. It is cut into pieces of 6 inches each. How many pieces are made?",
        "answer": "8",
    },
    {
        "id": "q11",
        "category": "multi_step",
        "question": "A book costs 12 dollars. A student buys 3 books and then uses a coupon for 6 dollars off the total. How much does the student pay?",
        "answer": "30",
    },
    {
        "id": "q12",
        "category": "comparison",
        "question": "Lena scored 14 points in the first game and 9 points in the second game. How many more points did she score in the first game than in the second?",
        "answer": "5",
    },
]


# -------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------

def load_questions(path: Optional[str], demo: bool = False) -> List[Dict[str, str]]:
    questions = DEFAULT_QUESTIONS
    if path:
        with open(path, "r", encoding="utf-8") as f:
            questions = json.load(f)
    if demo:
        return questions[:5]
    return questions


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def extract_final_answer(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]

    for line in reversed(lines):
        if line.lower().startswith("final answer:"):
            return line.split(":", 1)[1].strip()

    # numeric fallback: last integer or decimal in output
    matches = re.findall(r"-?\d+(?:\.\d+)?", output)
    if matches:
        return matches[-1]

    # yes/no fallback
    low = output.lower()
    if "yes" in low and "no" not in low:
        return "Yes"
    if "no" in low and "yes" not in low:
        return "No"

    return lines[-1] if lines else ""


def is_correct(predicted: str, gold: str) -> bool:
    gold_canonical = canonicalize_answer(gold)
    predicted_canonical = canonicalize_answer(predicted)

    # direct match
    if predicted_canonical == gold_canonical:
        return True

    # numeric fallback: check if gold number appears anywhere
    if re.fullmatch(r"-?\d+(?:\.\d+)?", gold_canonical):
        return re.search(rf"\b{re.escape(gold_canonical)}\b", str(predicted)) is not None

    return False


def tag_error(gold: str, predicted: str, raw_output: str) -> str:
    if is_correct(predicted, gold):
        return "correct"

    pred_norm = normalize_text(predicted)
    gold_norm = normalize_text(gold)

    numeric_gold = bool(re.search(r"\d", gold))
    numeric_pred = bool(re.search(r"\d", predicted))

    if numeric_gold and numeric_pred:
        return "calculation_error"

    if "final answer" not in raw_output.lower():
        return "format_error"

    if any(marker in raw_output.lower() for marker in ["step", "first", "then", "therefore"]):
        return "reasoning_error"

    if not predicted.strip():
        return "missing_answer"

    return "other_error"


def canonicalize_answer(text: str) -> str:
    text = str(text).strip()

    if text.lower().startswith("final answer:"):
        text = text.split(":", 1)[1].strip()

    low = text.lower()
    if re.search(r"\byes\b", low) and not re.search(r"\bno\b", low):
        return "yes"
    if re.search(r"\bno\b", low) and not re.search(r"\byes\b", low):
        return "no"

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        num = nums[-1]
        try:
            value = float(num)
            if value.is_integer():
                return str(int(value))
            return str(value)
        except ValueError:
            pass

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return normalize_text(text)


# -------------------------------------------------------------------
# OpenAI-compatible client
# -------------------------------------------------------------------

class ChatClient:
    def __init__(self, api_base: str, model: str, api_key: str = "", timeout: int = 120):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, str]], temperature: float) -> str:
        url = self.api_base + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def check_connection(self) -> bool:
        try:
            url = self.api_base + "/models"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return True
        except Exception:
            pass

        try:
            # fallback quick test against chat endpoint with a tiny request
            self.chat([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Reply with OK."},
            ], temperature=0.0)
            return True
        except Exception:
            return False


# -------------------------------------------------------------------
# Prompt builders
# -------------------------------------------------------------------

def build_messages(method: str, question: str) -> List[Dict[str, str]]:
    if method == "standard":
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful reasoning assistant. Solve the problem and output only one line: "
                    "Final answer: <answer>."
                ),
            },
            {"role": "user", "content": f"Question: {question}"},
        ]

    if method in {"cot", "cot_self_consistency"}:
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful reasoning assistant. Think step by step before answering. "
                    "End with exactly one line formatted as: Final answer: <answer>."
                ),
            },
            {"role": "user", "content": f"Question: {question}\nLet's think step by step."},
        ]

    raise ValueError(f"Unknown method: {method}")


# -------------------------------------------------------------------
# Experiment core
# -------------------------------------------------------------------

def run_self_consistency(
    client: ChatClient,
    question: str,
    samples: int,
    temperature: float,
    sleep_s: float,
) -> Tuple[str, List[Dict[str, str]]]:
    collected = []
    votes = []

    for _ in range(samples):
        messages = build_messages("cot_self_consistency", question)
        raw = client.chat(messages, temperature=temperature)
        final_answer = extract_final_answer(raw)
        collected.append({"raw_output": raw, "final_answer": final_answer})
        votes.append(normalize_text(final_answer))
        if sleep_s > 0:
            time.sleep(sleep_s)

    winner_norm, _ = Counter(votes).most_common(1)[0]
    winner_original = next(
        (sample["final_answer"] for sample in collected if normalize_text(sample["final_answer"]) == winner_norm),
        winner_norm,
    )
    return winner_original, collected


def run_experiment(
    client: ChatClient,
    questions: List[Dict[str, str]],
    self_consistency_samples: int,
    output_dir: str,
    demo: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    methods = ["standard", "cot", "cot_self_consistency"]
    raw_rows = []
    question_rows = []

    print("Running prompting experiment...\n")

    for idx, item in enumerate(questions, start=1):
        print(f"[{idx}/{len(questions)}] {item['id']} | {item['category']} | {item['question'][:72]}...")

        for method in methods:
            print(f"  -> method: {method}")

            if method == "standard":
                raw_output = client.chat(build_messages(method, item["question"]), temperature=0.0)
                predicted_answer = extract_final_answer(raw_output)
                samples = None

            elif method == "cot":
                raw_output = client.chat(build_messages(method, item["question"]), temperature=0.0)
                predicted_answer = extract_final_answer(raw_output)
                samples = None

            elif method == "cot_self_consistency":
                predicted_answer, samples = run_self_consistency(
                    client=client,
                    question=item["question"],
                    samples=self_consistency_samples,
                    temperature=0.7,
                    sleep_s=0.15 if not demo else 0.0,
                )
                raw_output = "\n\n--- SAMPLE SEPARATOR ---\n\n".join(sample["raw_output"] for sample in samples)

            else:
                raise ValueError(method)

            correct = is_correct(predicted_answer, item["answer"])
            error_type = tag_error(item["answer"], predicted_answer, raw_output)

            raw_rows.append({
                "question_id": item["id"],
                "category": item["category"],
                "method": method,
                "question": item["question"],
                "gold_answer": item["answer"],
                "predicted_answer": predicted_answer,
                "correct": correct,
                "error_type": error_type,
                "raw_output": raw_output,
                "samples": samples,
            })

            question_rows.append({
                "question_id": item["id"],
                "category": item["category"],
                "method": method,
                "question": item["question"],
                "gold_answer": item["answer"],
                "predicted_answer": predicted_answer,
                "correct": correct,
                "error_type": error_type,
            })

            print(f"     predicted={predicted_answer!r} | gold={item['answer']!r} | correct={correct}")

    summary_rows = build_summary(question_rows)
    save_outputs(output_dir, raw_rows, question_rows, summary_rows)
    make_accuracy_figure(summary_rows, os.path.join(output_dir, "figure_accuracy_by_method.png"))
    make_error_figure(summary_rows, os.path.join(output_dir, "figure_error_types_by_method.png"))
    save_summary_report(question_rows, summary_rows, os.path.join(output_dir, "summary_report.txt"))
    return raw_rows, question_rows, summary_rows


# -------------------------------------------------------------------
# Summaries and saving
# -------------------------------------------------------------------

def build_summary(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary_rows = []
    for method, items in grouped.items():
        total = len(items)
        correct_n = sum(1 for item in items if item["correct"])
        accuracy = correct_n / total if total else 0.0
        error_counter = Counter(item["error_type"] for item in items if not item["correct"])

        summary_rows.append({
            "method": method,
            "n_questions": total,
            "n_correct": correct_n,
            "accuracy": round(accuracy, 3),
            "calculation_error": error_counter.get("calculation_error", 0),
            "reasoning_error": error_counter.get("reasoning_error", 0),
            "format_error": error_counter.get("format_error", 0),
            "missing_answer": error_counter.get("missing_answer", 0),
            "other_error": error_counter.get("other_error", 0),
        })

    order = {"standard": 0, "cot": 1, "cot_self_consistency": 2}
    summary_rows.sort(key=lambda row: order.get(row["method"], 999))
    return summary_rows


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def save_outputs(output_dir: str, raw_rows: List[Dict], question_rows: List[Dict], summary_rows: List[Dict]) -> None:
    save_jsonl(os.path.join(output_dir, "raw_outputs.jsonl"), raw_rows)
    save_csv(
        os.path.join(output_dir, "question_results.csv"),
        question_rows,
        [
            "question_id",
            "category",
            "method",
            "question",
            "gold_answer",
            "predicted_answer",
            "correct",
            "error_type",
        ],
    )
    save_csv(
        os.path.join(output_dir, "summary_results.csv"),
        summary_rows,
        [
            "method",
            "n_questions",
            "n_correct",
            "accuracy",
            "calculation_error",
            "reasoning_error",
            "format_error",
            "missing_answer",
            "other_error",
        ],
    )


def save_summary_report(question_rows: List[Dict], summary_rows: List[Dict], path: str) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append("CHAIN-OF-THOUGHT EXPERIMENT SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    for row in summary_rows:
        lines.append(
            f"{row['method']}: accuracy={row['accuracy']:.3f}, "
            f"correct={row['n_correct']}/{row['n_questions']}"
        )
    lines.append("")

    lines.append("=" * 60)
    lines.append("PER-QUESTION RESULTS")
    lines.append("=" * 60)
    lines.append("")

    by_question = defaultdict(list)
    for row in question_rows:
        by_question[row["question_id"]].append(row)

    for qid in sorted(by_question.keys()):
        items = by_question[qid]
        prompt = items[0]["question"]
        gold = items[0]["gold_answer"]
        lines.append(f"[{qid}] {prompt}")
        lines.append(f"Gold answer: {gold}")
        for item in sorted(items, key=lambda x: {"standard": 0, "cot": 1, "cot_self_consistency": 2}[x["method"]]):
            lines.append(
                f"  {item['method']}: predicted={item['predicted_answer']} | "
                f"correct={item['correct']} | error_type={item['error_type']}"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------

def make_accuracy_figure(summary_rows: List[Dict], path: str) -> None:
    methods = [row["method"] for row in summary_rows]
    accuracies = [row["accuracy"] for row in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, accuracies)
    plt.ylim(0, 1)
    plt.xlabel("Method")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Prompting Method")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_error_figure(summary_rows: List[Dict], path: str) -> None:
    methods = [row["method"] for row in summary_rows]
    calc = [row["calculation_error"] for row in summary_rows]
    reasoning = [row["reasoning_error"] for row in summary_rows]
    format_err = [row["format_error"] for row in summary_rows]
    missing = [row["missing_answer"] for row in summary_rows]
    other = [row["other_error"] for row in summary_rows]

    x = list(range(len(methods)))
    width = 0.65

    plt.figure(figsize=(9, 5))
    plt.bar(x, calc, width, label="Calculation")
    bottom_1 = calc
    plt.bar(x, reasoning, width, bottom=bottom_1, label="Reasoning")
    bottom_2 = [a + b for a, b in zip(calc, reasoning)]
    plt.bar(x, format_err, width, bottom=bottom_2, label="Format")
    bottom_3 = [a + b + c for a, b, c in zip(calc, reasoning, format_err)]
    plt.bar(x, missing, width, bottom=bottom_3, label="Missing answer")
    bottom_4 = [a + b + c + d for a, b, c, d in zip(calc, reasoning, format_err, missing)]
    plt.bar(x, other, width, bottom=bottom_4, label="Other")

    plt.xticks(x, methods)
    plt.xlabel("Method")
    plt.ylabel("Count")
    plt.title("Error Types by Prompting Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CoT Experiment Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "mistral"),
        help="Model name for the chat endpoint",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key if required by your endpoint",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Directory for outputs",
    )
    parser.add_argument(
        "--questions",
        default=None,
        help="Optional JSON file with questions",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of self-consistency samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run only the first 5 questions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    ensure_output_dir(args.output)

    client = ChatClient(
        api_base=args.api_base,
        model=args.model,
        api_key=args.api_key,
    )

    print("Checking model connection...", end=" ", flush=True)
    if not client.check_connection():
        print("FAILED")
        print("Could not connect to the chat endpoint.")
        print(f"API base tried: {args.api_base}")
        raise SystemExit(1)
    print("OK")

    questions = load_questions(args.questions, demo=args.demo)
    print(f"Questions loaded: {len(questions)}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output}\n")

    _, _, summary_rows = run_experiment(
        client=client,
        questions=questions,
        self_consistency_samples=args.samples,
        output_dir=args.output,
        demo=args.demo,
    )

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    for row in summary_rows:
        print(
            f"{row['method']:22s} accuracy={row['accuracy']:.3f} "
            f"correct={row['n_correct']}/{row['n_questions']}"
        )

    print("\nSaved files:")
    print(f"  {os.path.join(args.output, 'raw_outputs.jsonl')}")
    print(f"  {os.path.join(args.output, 'question_results.csv')}")
    print(f"  {os.path.join(args.output, 'summary_results.csv')}")
    print(f"  {os.path.join(args.output, 'summary_report.txt')}")
    print(f"  {os.path.join(args.output, 'figure_accuracy_by_method.png')}")
    print(f"  {os.path.join(args.output, 'figure_error_types_by_method.png')}")


if __name__ == "__main__":
    main()

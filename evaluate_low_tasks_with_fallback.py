#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

DEFAULT_LOW_SCORE_TASKS = [
    "Background knowledge Image QA",
    # "Camera Status Classification",
    # "Camera Status Switching",
    "Commentary Generation",
    "Commentary Relevant QA",
    # "Game State Relevant QA",
    # "Jersey Number Recognition",
    # "Match Events and Statistical QA",
    # "Multi-view Foul Recognition",
    # "Replay Grounding",
]

COLOR_WORDS = {
    "black", "white", "red", "blue", "yellow", "green", "orange", "purple",
    "pink", "magenta", "gold", "silver", "navy", "brown", "gray", "grey",
    "maroon", "lavender", "cyan"
}


def extract_option(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    pattern = r"\b(O[1-9])\b"
    matches = re.findall(pattern, text.upper())
    return matches[-1] if matches else None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def collect_options(item: Dict) -> List[Tuple[str, str]]:
    options = []
    for i in range(1, 10):
        key = f"O{i}"
        val = item.get(key)
        if val is not None and str(val).strip() != "":
            options.append((key, str(val)))
    return options


def _score_option(question: str, option_text: str, model_reply: str, task: str) -> float:
    q = normalize_text(question)
    o = normalize_text(option_text)
    r = normalize_text(model_reply)

    score = 0.0

    # 1) lexical overlap with question + reply
    q_tokens = set(tokenize(q))
    o_tokens = set(tokenize(o))
    r_tokens = set(tokenize(r))

    score += 1.0 * len(o_tokens & q_tokens)
    score += 1.5 * len(o_tokens & r_tokens)

    # 2) phrase-hit bonus
    if o and o in r:
        score += 4.0

    # 3) number matching bonus
    nums_qr = set(re.findall(r"\b\d+\b", q + " " + r))
    nums_o = set(re.findall(r"\b\d+\b", o))
    if nums_qr and nums_o and (nums_qr & nums_o):
        score += 3.0

    # 4) color matching bonus
    if "jersey" in q or "color" in q or "kit" in q:
        colors_in_qr = {w for w in tokenize(q + " " + r) if w in COLOR_WORDS}
        colors_in_o = {w for w in tokenize(o) if w in COLOR_WORDS}
        if colors_in_qr and colors_in_o and (colors_in_qr & colors_in_o):
            score += 3.0

    # 5) task-specific tiny prior
    task_l = normalize_text(task)
    if "commentary" in task_l:
        # Commentary options often are sentences; prefer richer textual options
        score += min(len(o_tokens) / 50.0, 0.8)
    elif "jersey number" in task_l:
        if re.search(r"^\d+$", o):
            score += 0.8

    return score


def non_random_fallback(item: Dict, model_reply: Optional[str]) -> str:
    question = str(item.get("Q", ""))
    task = str(item.get("task", ""))
    options = collect_options(item)

    if not options:
        return "O1"

    reply = model_reply or ""

    scored = []
    for key, option_text in options:
        score = _score_option(question, option_text, reply, task)
        scored.append((key, score))

    # deterministic tie-break by option index (O1 < O2 < ...)
    scored.sort(key=lambda x: (-x[1], int(x[0][1:])))
    return scored[0][0]


def filter_first_200_low_tasks(data: List[Dict], low_tasks: List[str]) -> List[Dict]:
    low_task_set = {t.strip() for t in low_tasks if t.strip()}
    first_200 = data[:200]
    return [x for x in first_200 if str(x.get("task", "")).strip() in low_task_set]


def run(
    input_file: str,
    materials_folder: str,
    api_key: str,
    low_tasks: List[str],
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON root must be a list.")

    filtered = filter_first_200_low_tasks(data, low_tasks)
    print(f"Filtered {len(filtered)} low-score-task items from first 200.")

    task_dist = Counter([str(x.get("task", "Unknown")) for x in filtered])
    print("Filtered task distribution:")
    for t, c in task_dist.items():
        print(f"  - {t}: {c}")

    os.environ["DEEPSEEK_API_KEY"] = api_key
    try:
        from multiagent_platform import EXECUTE_TOOL_CHAIN
    except ImportError as e:
        raise RuntimeError(f"Failed to import EXECUTE_TOOL_CHAIN: {e}")

    total_processed = 0
    total_correct = 0
    task_stats: Dict[str, Dict[str, int]] = {}
    fallback_used_total = 0

    for item in tqdm(filtered, desc="Predicting low-score tasks"):
        qid = item.get("id")
        question = item.get("Q", "")
        task = str(item.get("task", "Unknown"))
        closeA = str(item.get("closeA", ""))

        options_text = ""
        option_count = 0
        for i in range(1, 10):
            key = f"O{i}"
            if key in item and item[key]:
                options_text += f"{key}: {item[key]}\n"
                option_count += 1

        old_materials = item.get("materials", [])
        materials = []
        if old_materials:
            for m in old_materials:
                materials.append(os.path.join(materials_folder, m))

        material_dict = {"materials": materials} if materials else None

        raw_reply = ""
        pred = None
        used_fallback = False

        try:
            raw_reply = EXECUTE_TOOL_CHAIN(
                query=question,
                material=material_dict,
                options=options_text,
            )
            pred = extract_option(raw_reply)
        except Exception as e:
            raw_reply = f"[ERROR] {e}"
            pred = None

        if not pred:
            pred = non_random_fallback(item, raw_reply)
            used_fallback = True

        if not pred and option_count > 0:
            # final deterministic guard, still non-random
            pred = "O1"

        is_correct = bool(pred and closeA and pred.upper() == closeA.upper())

        if task not in task_stats:
            task_stats[task] = {"total": 0, "correct": 0}

        total_processed += 1
        task_stats[task]["total"] += 1
        if is_correct:
            total_correct += 1
            task_stats[task]["correct"] += 1

        if used_fallback:
            fallback_used_total += 1

        print(
            f"ID: {qid} | Task: {task} | Pred: {pred} | GT: {closeA} | "
            f"{'CORRECT' if is_correct else 'WRONG'}"
        )

    print("\n" + "=" * 50)
    print("LOW-TASK EVALUATION REPORT")
    print("=" * 50)
    overall_acc = (total_correct / total_processed * 100) if total_processed > 0 else 0.0
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_processed})")
    print(f"Non-random fallback used: {fallback_used_total}/{total_processed}")

    print("\nAccuracy by Task Type:")
    print("-" * 80)
    print(f"{'Task Type':<40} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<10}")
    print("-" * 80)
    for t_type, stats in sorted(task_stats.items()):
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
        print(f"{t_type:<40} | {stats['total']:<6} | {stats['correct']:<8} | {acc:6.2f}%")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Filter low-score tasks from first 200 test items and apply non-random fallback."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/root/autodl-tmp/SoccerNet_Challenge_VQA/test/test_with_tasks.json",
        help="Input JSON with task labels",
    )
    parser.add_argument(
        "--materials_folder",
        type=str,
        default="/root/autodl-tmp/SoccerNet_Challenge_VQA/test",
        help="Base folder for materials",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="DeepSeek API key used by multiagent_platform",
    )
    parser.add_argument(
        "--low_tasks",
        type=str,
        default=",".join(DEFAULT_LOW_SCORE_TASKS),
        help="Comma-separated low-score tasks",
    )

    args = parser.parse_args()
    low_tasks = [x.strip() for x in args.low_tasks.split(",") if x.strip()]

    run(
        input_file=args.input_file,
        materials_folder=args.materials_folder,
        api_key=args.api_key,
        low_tasks=low_tasks,
    )


if __name__ == "__main__":
    main()

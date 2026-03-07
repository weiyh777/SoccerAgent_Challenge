#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

DEFAULT_TASKS = [
    "Background knowledge text QA",
    "Match Situation QA",
    "Match Events and Statistical QA",
    "Camera Status Classification",
    "Background knowledge Image QA",
    "Jersey Number Recognition",
    "Score and Time Relevant QA",
    "Game State Relevant QA",
    "Camera Status Switching",
    "Replay Grounding",
    "Action Classification",
    "Commentary Generation",
    "Commentary Relevant QA",
    "Jersy Color Relevant QA",
    "Multi-view Foul Recognition",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_task_names(tasks_csv: Optional[str]) -> List[str]:
    if not tasks_csv or not os.path.exists(tasks_csv):
        return DEFAULT_TASKS

    task_names = []
    with open(tasks_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, quotechar='"', skipinitialspace=True)
        for row in reader:
            task = (row.get("task") or "").strip().strip('"')
            if task:
                task_names.append(task)

    return task_names or DEFAULT_TASKS


def pick_task(task_name: str, available_tasks: List[str]) -> str:
    if task_name in available_tasks:
        return task_name

    lower_map = {t.lower(): t for t in available_tasks}
    return lower_map.get(task_name.lower(), task_name)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def has_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def inspect_materials(materials: Optional[List[str]]) -> Dict[str, int]:
    info = {"total": 0, "images": 0, "videos": 0, "dirs_or_unknown": 0}
    if not materials:
        return info

    for m in materials:
        info["total"] += 1
        ext = os.path.splitext(str(m))[1].lower()
        if ext in IMAGE_EXTS:
            info["images"] += 1
        elif ext in VIDEO_EXTS:
            info["videos"] += 1
        else:
            info["dirs_or_unknown"] += 1
    return info


def classify_task(item: Dict, available_tasks: List[str]) -> str:
    question = normalize_text(str(item.get("Q", "")))
    materials = item.get("materials") or []
    media = inspect_materials(materials)
    has_material = media["total"] > 0

    if has_any(question, [r"multi[- ]view", r"different angles", r"from multiple views"]) or (
        has_any(question, [r"foul"]) and media["videos"] >= 2
    ):
        return pick_task("Multi-view Foul Recognition", available_tasks)

    if has_any(question, [r"camera.*switch", r"switching", r"shot change", r"camera transition"]):
        return pick_task("Camera Status Switching", available_tasks)

    if has_any(question, [r"camera", r"lens", r"camera position", r"camera status"]):
        return pick_task("Camera Status Classification", available_tasks)

    if has_any(question, [r"replay", r"being replayed", r"which clip"]):
        return pick_task("Replay Grounding", available_tasks)

    if has_any(question, [r"write (a )?commentary", r"generate commentary", r"commentary text", r"commentary content"]):
        return pick_task("Commentary Generation", available_tasks)

    if has_any(question, [r"jersey number", r"shirt.*number", r"number displayed on.*shirt", r"number on the player"]):
        return pick_task("Jersey Number Recognition", available_tasks)

    if has_any(question, [r"jersey color", r"color jersey", r"kit color", r"wearing\?", r"which color jersey"]):
        return pick_task("Jersy Color Relevant QA", available_tasks)

    if has_any(question, [r"score", r"game time", r"match time", r"minute", r"stoppage time", r"time shown", r"from this image, what is the score"]):
        return pick_task("Score and Time Relevant QA", available_tasks)

    if has_any(question, [r"how many players", r"formation", r"position of the players", r"tactical", r"game state", r"visible in this image"]):
        return pick_task("Game State Relevant QA", available_tasks)

    if has_any(question, [r"classify", r"action category", r"what kind of football event", r"appropriate category", r"football event in this video"]):
        return pick_task("Action Classification", available_tasks)

    if has_any(question, [r"how many corners", r"first half", r"second half", r"shots on target", r"possession", r"statistics"]):
        return pick_task("Match Events and Statistical QA", available_tasks)

    if has_any(question, [r"assist", r"goal", r"red card", r"yellow card", r"head coach", r"against", r"draw against", r"during their"]):
        return pick_task("Match Situation QA", available_tasks)

    bio_or_wiki = has_any(
        question,
        [
            r"date of birth",
            r"which club did",
            r"joined",
            r"youth clubs?",
            r"youth teams?",
            r"nickname",
            r"appearances",
            r"national team",
            r"represent",
            r"who was .* head coach",
            r"which team did",
        ],
    )

    if has_material and bio_or_wiki:
        if has_any(question, [r"in this video", r"from this image", r"this player", r"commentary", r"who headed", r"who .* in this video"]):
            return pick_task("Commentary Relevant QA", available_tasks)
        return pick_task("Background knowledge Image QA", available_tasks)

    if has_material:
        return pick_task("Background knowledge Image QA", available_tasks)

    return pick_task("Background knowledge text QA", available_tasks)


def infer_tasks(
    input_file: str,
    output_file: str,
    tasks_csv: Optional[str],
    task_field: str,
    keep_existing: bool,
    also_update_task_type: bool,
) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON root must be a list.")

    available_tasks = load_task_names(tasks_csv)

    counts = Counter()
    for item in data:
        if not isinstance(item, dict):
            continue

        existing = str(item.get(task_field, "")).strip()
        if keep_existing and existing and existing.lower() != "none":
            task = existing
        else:
            task = classify_task(item, available_tasks)

        item[task_field] = task
        if also_update_task_type:
            item["task type"] = task
        counts[task] += 1

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Done. Saved to: {output_file}")
    print("Task distribution:")
    for task_name, cnt in counts.most_common():
        print(f"  - {task_name}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="Infer task labels from SoccerNet VQA questions.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge.json",
        help="Path to input JSON",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/root/autodl-tmp/SoccerAgent/challenge_with_tasks.json",
        help="Path to output JSON",
    )
    parser.add_argument(
        "--tasks_csv",
        type=str,
        default="/root/autodl-tmp/SoccerAgent/tasks.csv",
        help="Path to tasks.csv",
    )
    parser.add_argument(
        "--task_field",
        type=str,
        default="task",
        help="Field name used to write inferred task labels",
    )
    parser.add_argument(
        "--keep_existing",
        action="store_true",
        default=False,
        help="Keep existing non-empty task field values",
    )
    parser.add_argument(
        "--also_update_task_type",
        action="store_true",
        default=False,
        help="Also write inferred result into 'task type' field",
    )

    args = parser.parse_args()

    infer_tasks(
        input_file=args.input_file,
        output_file=args.output_file,
        tasks_csv=args.tasks_csv,
        task_field=args.task_field,
        keep_existing=args.keep_existing,
        also_update_task_type=args.also_update_task_type,
    )


if __name__ == "__main__":
    main()

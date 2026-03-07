#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SoccerNet VQA Challenge 2026 - 运行脚本
使用此脚本生成Challenge提交结果

用法:
    python run_challenge_submission.py --mode baseline --model qwen --model_path <path>
    python run_challenge_submission.py --mode agent_lite --api_key <deepseek_api_key>
    python run_challenge_submission.py --mode agent --task_list "Camera Status Classification,Replay Grounding"
"""

import json
import os
import re
import argparse
import difflib
from tqdm import tqdm

os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"


def extract_option(text):
    """从文本中提取选项答案（优先最终答案区域，避免被Query里的O1/O2污染）"""
    if not text:
        return None

    raw = str(text)

    # 1) 优先匹配显式答案行: Answer: O2 / Answer: Option O2
    answer_line_matches = re.findall(r"(?im)^\s*answer\s*:\s*(?:option\s*)?(o[1-9])\b", raw)
    if answer_line_matches:
        return answer_line_matches[-1].upper()

    # 2) 对于agent trace，优先仅在 </EndCall> 之后搜索，避免匹配到Query中的选项列表
    tail = raw.split("</EndCall>")[-1] if "</EndCall>" in raw else raw
    tail_matches = re.findall(r"\b(O[1-9])\b", tail.upper())
    if tail_matches:
        return tail_matches[-1]

    # 3) 全文只有唯一一个选项时才兜底返回
    all_matches = re.findall(r"\b(O[1-9])\b", raw.upper())
    if len(set(all_matches)) == 1 and all_matches:
        return all_matches[0]

    return None


COLOR_WORDS = {
    "black", "white", "red", "blue", "yellow", "green", "orange", "purple",
    "pink", "magenta", "gold", "silver", "navy", "brown", "gray", "grey",
    "maroon", "lavender", "cyan"
}


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def collect_options(item):
    options = []
    for i in range(1, 10):
        key = f"O{i}"
        val = item.get(key)
        if val is not None and str(val).strip() != "":
            options.append((key, str(val)))
    return options


def _score_option(question, option_text, model_reply, task):
    q = normalize_text(question)
    o = normalize_text(option_text)
    r = normalize_text(model_reply)

    score = 0.0

    q_tokens = set(tokenize(q))
    o_tokens = set(tokenize(o))
    r_tokens = set(tokenize(r))

    score += 1.0 * len(o_tokens & q_tokens)
    score += 1.5 * len(o_tokens & r_tokens)

    if o and o in r:
        score += 4.0

    nums_qr = set(re.findall(r"\b\d+\b", q + " " + r))
    nums_o = set(re.findall(r"\b\d+\b", o))
    if nums_qr and nums_o and (nums_qr & nums_o):
        score += 3.0

    if "jersey" in q or "color" in q or "kit" in q:
        colors_in_qr = {w for w in tokenize(q + " " + r) if w in COLOR_WORDS}
        colors_in_o = {w for w in tokenize(o) if w in COLOR_WORDS}
        if colors_in_qr and colors_in_o and (colors_in_qr & colors_in_o):
            score += 3.0

    task_l = normalize_text(task)
    if "commentary" in task_l:
        score += min(len(o_tokens) / 50.0, 0.8)
    elif "jersey number" in task_l and re.search(r"^\d+$", o):
        score += 0.8

    return score


def _extract_decision_text(model_reply):
    reply = str(model_reply or "")

    if "</EndCall>" in reply:
        tail = reply.split("</EndCall>")[-1].strip()
        if tail:
            return tail

    answer_lines = re.findall(r"(?im)^\s*answer\s*:\s*(.+)$", reply)
    if answer_lines:
        return answer_lines[-1].strip()

    return reply


def _match_option_from_decision(options, decision_text):
    decision_text = str(decision_text or "")
    norm_decision = normalize_text(decision_text)
    if not norm_decision:
        return None

    answer_lines = re.findall(r"(?im)^\s*answer\s*:\s*(.+)$", decision_text)
    focused_answer = normalize_text(answer_lines[-1]) if answer_lines else norm_decision

    exact = [key for key, option_text in options if normalize_text(option_text) == focused_answer]
    if len(exact) == 1:
        return exact[0]

    contains = [
        key for key, option_text in options
        if normalize_text(option_text) and normalize_text(option_text) in focused_answer
    ]
    if len(contains) == 1:
        return contains[0]

    return None


def _semantic_map_option_from_decision(options, decision_text):
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "to", "from", "of", "and", "or", "in", "on", "for", "with",
        "camera", "viewpoint", "option", "answer", "this", "that", "it", "be", "as", "by", "at", "does", "did", "not"
    }

    decision_text = str(decision_text or "")
    norm_decision = normalize_text(decision_text)
    if not norm_decision:
        return None, "Semantic mapping skipped: empty decision text."

    answer_lines = re.findall(r"(?im)^\s*answer\s*:\s*(.+)$", decision_text)
    focused_answer = normalize_text(answer_lines[-1]) if answer_lines else norm_decision

    def _core_tokens(text):
        return {tok for tok in tokenize(text) if tok not in stopwords}

    ans_tokens = _core_tokens(focused_answer)
    scored = []

    for key, option_text in options:
        opt_norm = normalize_text(option_text)
        if not opt_norm:
            continue

        opt_tokens = _core_tokens(opt_norm)
        overlap = len(ans_tokens & opt_tokens)

        score = 0.0
        if opt_norm in focused_answer or focused_answer in opt_norm:
            score += 3.0

        if opt_tokens:
            score += 1.6 * (overlap / len(opt_tokens))
        if ans_tokens:
            score += 1.2 * (overlap / len(ans_tokens))

        score += 0.8 * difflib.SequenceMatcher(None, focused_answer, opt_norm).ratio()
        scored.append((key, option_text, score))

    if not scored:
        return None, "Semantic mapping skipped: no valid options."

    scored.sort(key=lambda x: (-x[2], int(x[0][1:])))
    best_key, best_text, best_score = scored[0]
    second_score = scored[1][2] if len(scored) > 1 else 0.0
    margin = best_score - second_score

    trace = f"Semantic mapping top={best_key} score={best_score:.3f} margin={margin:.3f} text={best_text}"

    if best_score >= 1.25 and margin >= 0.18:
        return best_key, trace

    return None, trace + " | below confidence threshold"


def non_random_fallback(item, model_reply):
    question = str(item.get("Q", ""))
    task = str(item.get("task", ""))
    options = collect_options(item)

    if not options:
        trace = "No options found. Use deterministic guard O1."
        return "O1", trace

    reply = model_reply or ""
    decision_text = _extract_decision_text(reply)

    direct = _match_option_from_decision(options, decision_text)
    if direct:
        lines = [
            "[Non-random fallback triggered]",
            f"Question: {question}",
            f"Task: {task}",
            f"Decision text used: {decision_text}",
            f"Chosen fallback answer (direct decision match): {direct}",
        ]
        return direct, "\n".join(lines)

    semantic_key, semantic_trace = _semantic_map_option_from_decision(options, decision_text)
    if semantic_key:
        lines = [
            "[Non-random fallback triggered]",
            f"Question: {question}",
            f"Task: {task}",
            f"Decision text used: {decision_text}",
            semantic_trace,
            f"Chosen fallback answer (semantic decision match): {semantic_key}",
        ]
        return semantic_key, "\n".join(lines)

    scored = []
    for key, option_text in options:
        score = _score_option(question, option_text, decision_text, task)
        scored.append((key, option_text, score))

    scored.sort(key=lambda x: (-x[2], int(x[0][1:])))
    best = scored[0][0]

    lines = [
        "[Non-random fallback triggered]",
        f"Question: {question}",
        f"Task: {task}",
        f"Decision text used: {decision_text}",
        semantic_trace,
        f"Raw reply: {reply}",
        "Option scores:",
    ]
    for key, option_text, score in scored:
        lines.append(f"- {key} | score={score:.2f} | text={option_text}")
    lines.append(f"Chosen fallback answer: {best}")
    return best, "\n".join(lines)


def parse_task_list(task_list_str):
    if not task_list_str:
        return []
    return [x.strip() for x in task_list_str.split(",") if x.strip()]


def parse_id_list(id_list_str):
    if not id_list_str:
        return []
    ids = []
    for value in id_list_str.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            ids.append(int(value))
        except ValueError:
            print(f"Warning: invalid id in --id_list: {value}")
    return ids


def filter_data_by_tasks_and_ids(data, task_list=None, id_list=None, task_field="task"):
    task_list = task_list or []
    id_list = id_list or []

    if not task_list and not id_list:
        return data

    task_set = set(task_list)
    id_set = set(id_list)

    filtered = []
    for item in data:
        match_task = bool(task_set) and str(item.get(task_field, "")).strip() in task_set

        item_id = item.get("id")
        try:
            item_id = int(item_id)
        except (TypeError, ValueError):
            pass

        match_id = bool(id_set) and item_id in id_set

        if match_task or match_id:
            filtered.append(item)

    if task_list:
        print(f"Task filter enabled ({task_field}): {task_list}")
    if id_list:
        print(f"ID filter enabled: {sorted(id_set)}")
    print(f"Questions after task/id union filter: {len(filtered)}/{len(data)}")
    return filtered


def run_baseline_qwen(input_file, output_file, materials_folder, model_path, task_list=None, id_list=None, task_field="task"):
    """使用Qwen2.5-VL baseline"""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"Loading Qwen2.5-VL from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    instruction = """You are a football expert. You are provided with a question 'Q' and multiple options.
Please answer the question with one option that best matches the question (reply with 'O1', 'O2', 'O3', 'O4'...).
Do not include any other text or explanations."""

    with open(input_file, 'r') as f:
        data = json.load(f)
    data = filter_data_by_tasks_and_ids(data, task_list=task_list, id_list=id_list, task_field=task_field)

    results = []
    for item in tqdm(data, desc="Processing questions"):
        question_id = item["id"]
        question = item["Q"]

        prompt = f"Q: {question}\n"
        option_count = 0
        for i in range(1, 10):
            opt = item.get(f"O{i}")
            if opt:
                prompt += f"O{i}: {opt}\n"
                option_count += 1

        old_materials = item.get("materials", [])
        materials = []
        if old_materials:
            for m in old_materials:
                full_path = os.path.join(materials_folder, m)
                if os.path.exists(full_path):
                    materials.append(full_path)
                elif os.path.isdir(full_path):
                    files = sorted([os.path.join(full_path, f) for f in os.listdir(full_path)])
                    materials.extend(files)

        reply = ""
        try:
            conversation = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": []}
            ]

            if materials:
                for m in materials:
                    ext = os.path.splitext(m)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                        conversation[1]["content"].append({"type": "image", "image": f"file://{m}"})
                    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                        conversation[1]["content"].append({
                            "type": "video",
                            "video": f"file://{m}",
                            "max_pixels": 360 * 640,
                            "fps": 1.0
                        })

            conversation[1]["content"].append({"type": "text", "text": prompt})

            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            output_ids = model.generate(**inputs, max_new_tokens=32)
            generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            reply = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            answer = extract_option(reply)

        except Exception as e:
            print(f"Error on question {question_id}: {e}")
            answer = None

        if not answer:
            answer, _ = non_random_fallback(item, reply)

        results.append({"id": question_id, "Answer": answer})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_file}")
    return results


def run_baseline_gpt(input_file, output_file, materials_folder, api_key, task_list=None, id_list=None, task_field="task"):
    """使用GPT-4o baseline"""
    from openai import OpenAI
    import base64

    client = OpenAI(api_key=api_key)

    instruction = """You are a football expert. You are provided with a question 'Q' and multiple options.
Please answer the question with one option that best matches the question (reply with 'O1', 'O2', 'O3', 'O4'...).
Do not include any other text or explanations."""

    def encode_image(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    with open(input_file, 'r') as f:
        data = json.load(f)
    data = filter_data_by_tasks_and_ids(data, task_list=task_list, id_list=id_list, task_field=task_field)

    results = []
    for item in tqdm(data, desc="Processing questions"):
        question_id = item["id"]
        question = item["Q"]

        prompt = f"Q: {question}\n"
        for i in range(1, 10):
            opt = item.get(f"O{i}")
            if opt:
                prompt += f"O{i}: {opt}\n"

        old_materials = item.get("materials", [])
        materials = []
        if old_materials:
            for m in old_materials:
                full_path = os.path.join(materials_folder, m)
                if os.path.exists(full_path):
                    materials.append(full_path)

        reply = ""
        try:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]

            for m in materials:
                ext = os.path.splitext(m)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    b64_image = encode_image(m)
                    messages[1]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                    })

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=32
            )
            reply = response.choices[0].message.content
            answer = extract_option(reply)

        except Exception as e:
            print(f"Error on question {question_id}: {e}")
            answer = None

        if not answer:
            answer, _ = non_random_fallback(item, reply)

        results.append({"id": question_id, "Answer": answer})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_file}")
    return results


def run_soccer_agent(input_file, output_file, materials_folder, api_key, task_list=None, id_list=None, task_field="task"):
    """
    使用SoccerAgent多智能体系统

    注意: 完整的SoccerAgent需要配置多个预训练模型:
    1. UniSoccer分类模型 (pretrained_classification.pth)
    2. UniSoccer解说模型 (downstream_commentary_all_open.pth)
    3. 其他工具的模型权重

    如果没有配置这些模型，请使用 --mode agent_lite 模式
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    os.environ['DEEPSEEK_API_KEY'] = api_key

    try:
        from multiagent_platform import EXECUTE_TOOL_CHAIN
    except ModuleNotFoundError as e:
        print(f"\n❌ Error importing SoccerAgent modules: {e}")
        print("\n⚠️  SoccerAgent requires several pretrained models to be configured:")
        print("   1. Edit toolbox/unisoccer/inference/distribution.py")
        print("   2. Set CHECKPOINT_PATH_CLASSIFICATION to your classification model path")
        print("   3. Set CHECKPOINT_PATH_COMMENTARY to your commentary model path")
        print("   4. Download models from: https://huggingface.co/Homie0609/UniSoccer")
        print("\n�� Alternative: Use --mode agent_lite for a simplified version")
        raise

    with open(input_file, 'r') as f:
        data = json.load(f)
    data = filter_data_by_tasks_and_ids(data, task_list=task_list, id_list=id_list, task_field=task_field)

    results = []
    for item in tqdm(data, desc="Processing with SoccerAgent"):
        question_id = item["id"]
        question = item["Q"]

        options_text = ""
        for i in range(1, 10):
            opt = item.get(f"O{i}")
            if opt:
                options_text += f"O{i}: {opt}\n"

        old_materials = item.get("materials", [])
        materials = []
        if old_materials:
            for m in old_materials:
                full_path = os.path.join(materials_folder, m)
                materials.append(full_path)

        material_dict = {"materials": materials} if materials else None

        fallback_trace = ""
        try:
            result = EXECUTE_TOOL_CHAIN(
                query=question,
                material=material_dict,
                options=options_text
            )
            answer = extract_option(result)

        except Exception as e:
            print(f"Error on question {question_id}: {e}")
            result = f"[ERROR] {e}"
            answer = None

        if not answer:
            answer, fallback_trace = non_random_fallback(item, result)

        thoughts_dir = os.path.join(os.path.dirname(output_file), "agent_thoughts_0222_camera")
        os.makedirs(thoughts_dir, exist_ok=True)
        with open(os.path.join(thoughts_dir, f"{question_id}.txt"), 'w', encoding='utf-8') as f:
            replan_events = re.findall(r"\[(?:RUNTIME_)?REPLAN_TRIGGERED\s+\d+/\d+\]", str(result or ""))
            if replan_events:
                f.write("[Replan Summary]\n")
                f.write(f"Replan triggers: {len(replan_events)}\n")
                f.write(f"Replan events: {'; '.join(replan_events)}\n")
                f.write("=" * 60 + "\n\n")

            f.write(str(result))
            if fallback_trace:
                f.write("\n\n" + "=" * 60 + "\n")
                f.write(fallback_trace)

        results.append({"id": question_id, "Answer": answer})
        print(f"Q{question_id}: {answer}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_file}")
    return results


def run_agent_lite(input_file, output_file, materials_folder, api_key, task_list=None, id_list=None, task_field="task"):
    """
    使用简化版Agent (仅依赖DeepSeek API，不需要本地模型)
    适合没有配置完整SoccerAgent环境的情况
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    task_prompt = """You are a soccer expert AI assistant. Analyze the question and provide the answer.

For different types of questions, use the following reasoning:
- If asking about player/team/referee background: Think about their career history
- If asking about match events/statistics: Consider the match context
- If asking about camera/video content: Describe what would be visible
- If asking about jersey numbers/colors: Focus on visual details
- If asking about scores/times: Look for scoreboard information
- If asking about actions in video: Consider common soccer events

Question: {question}

Options:
{options}

{material_note}

Think step by step, then provide your answer as ONLY the option number (O1, O2, O3, or O4)."""

    with open(input_file, 'r') as f:
        data = json.load(f)
    data = filter_data_by_tasks_and_ids(data, task_list=task_list, id_list=id_list, task_field=task_field)

    results = []
    for item in tqdm(data, desc="Processing with Agent-Lite"):
        question_id = item["id"]
        question = item["Q"]

        options_text = ""
        for i in range(1, 10):
            opt = item.get(f"O{i}")
            if opt:
                options_text += f"O{i}: {opt}\n"

        old_materials = item.get("materials", [])
        material_note = ""
        if old_materials:
            material_note = f"Note: This question includes visual materials: {old_materials}"

        prompt = task_prompt.format(
            question=question,
            options=options_text,
            material_note=material_note
        )

        reply = ""
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256
            )
            reply = response.choices[0].message.content
            answer = extract_option(reply)

        except Exception as e:
            print(f"Error on question {question_id}: {e}")
            answer = None

        if not answer:
            answer, _ = non_random_fallback(item, reply)

        results.append({"id": question_id, "Answer": answer})

        if question_id % 50 == 0:
            print(f"Processed {question_id} questions...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_file}")
    return results


def create_submission_zip(output_file, zip_path):
    """创建提交用的zip文件"""
    import zipfile
    import shutil

    metadata_path = os.path.join(os.path.dirname(output_file), "metadata.json")
    if output_file != metadata_path:
        shutil.copy(output_file, metadata_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(metadata_path, "metadata.json")

    print(f"Submission zip created: {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="SoccerNet VQA Challenge Runner")
    parser.add_argument('--mode', choices=['baseline', 'agent', 'agent_lite'], default='agent',
                        help='Running mode: baseline (VLM), agent (full SoccerAgent), or agent_lite (DeepSeek only)')
    parser.add_argument('--model', choices=['qwen', 'gpt'], default='qwen',
                        help='Model to use for baseline mode')
    parser.add_argument('--input_file', type=str,
                        default='/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge_with_tasks.json',
                        help='Input JSON file path')
    parser.add_argument('--output_file', type=str,
                        default='/root/autodl-tmp/SoccerAgent/agent_result_0222_new_camera.json',
                        help='Output JSON file path')
    parser.add_argument('--materials_folder', type=str,
                        default='/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge',
                        help='Materials folder path')
    parser.add_argument('--model_path', type=str, default="/root/autodl-tmp/SoccerAgent/model/Qwen2.5-VL-3B-Instruct",
                        help='Qwen model path (required for qwen mode)')
    parser.add_argument('--api_key', type=str, default="",
                        help='API key (OpenAI for gpt, DeepSeek for agent/agent_lite)')
    parser.add_argument('--create_zip', action='store_true', default=False,
                        help='Create submission zip after generating results')
    parser.add_argument('--task_list', type=str, default='',
                        help='Comma-separated task list. If provided, only questions in these tasks are processed.')
    parser.add_argument('--id_list', type=str, default='',
                        help='Comma-separated question IDs. These IDs are unioned with task_list filter.')
    parser.add_argument('--task_field', type=str, default='task',
                        help='Task field name in input file (e.g., task or task type)')

    args = parser.parse_args()

    print("=" * 60)
    print("SoccerNet VQA Challenge 2026 - Submission Generator")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Materials: {args.materials_folder}")
    print(f"Task Field: {args.task_field}")
    print("=" * 60)

    # task_list = parse_task_list(args.task_list)
    task_list = [
        # "Background knowledge text QA",
        # "Match Situation QA",
        # "Match Events and Statistical QA",
        "Camera Status Classification",
        # "Background knowledge Image QA",
        # "Jersey Number Recognition",
        # "Score and Time Relevant QA",
        # "Game State Relevant QA",
        "Camera Status Switching",
        # "Replay Grounding",
        # "Action Classification",
        # "Commentary Generation",
        # "Commentary Relevant QA",
        # "Jersy Color Relevant QA",
        # "Multi-view Foul Recognition",
    ]
    if task_list:
        print(f"Task List: {task_list}")

    id_list = parse_id_list(args.id_list)
    # id_list = "125,127,134,138,141,154,155,164,219,221,236,250,255,267,276,298,318,327,334,363,379,383,384,404,433,459,465,471,474,476,481,482,497"
    # id_list = parse_id_list(id_list)
    if id_list:
        print(f"ID List: {id_list}")

    if args.mode == 'baseline':
        if args.model == 'qwen':
            if not args.model_path:
                raise ValueError("--model_path is required for Qwen model")
            run_baseline_qwen(args.input_file, args.output_file,
                              args.materials_folder, args.model_path,
                              task_list=task_list, id_list=id_list, task_field=args.task_field)
        else:
            if not args.api_key:
                raise ValueError("--api_key is required for GPT model")
            run_baseline_gpt(args.input_file, args.output_file,
                             args.materials_folder, args.api_key,
                             task_list=task_list, id_list=id_list, task_field=args.task_field)
    elif args.mode == 'agent':
        if not args.api_key:
            raise ValueError("--api_key is required for SoccerAgent")
        run_soccer_agent(args.input_file, args.output_file,
                         args.materials_folder, args.api_key,
                         task_list=task_list, id_list=id_list, task_field=args.task_field)
    elif args.mode == 'agent_lite':
        if not args.api_key:
            raise ValueError("--api_key is required for Agent-Lite")
        run_agent_lite(args.input_file, args.output_file,
                       args.materials_folder, args.api_key,
                       task_list=task_list, id_list=id_list, task_field=args.task_field)

    if args.create_zip:
        zip_path = os.path.join(os.path.dirname(args.output_file), "submission.zip")
        create_submission_zip(args.output_file, zip_path)

    print("\nDone! Next steps:")
    print("1. Check the output file: " + args.output_file)
    print("2. Create submission.zip: zip submission.zip metadata.json")
    print("3. Upload to Codabench: https://www.codabench.org/competitions/11087/")


if __name__ == "__main__":
    main()

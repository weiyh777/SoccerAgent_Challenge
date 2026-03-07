#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import argparse
import sys
import random

# 添加当前目录到sys.path以导入multiagent_platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def extract_option(text):
    """从文本中提取选项答案"""
    import re
    pattern = r'\b(O[1-9])\b'
    matches = re.findall(pattern, text.upper())
    if matches:
        return matches[-1]
    return None

def main():
    parser = argparse.ArgumentParser(description="Test SoccerAgent on a specific task type")
    parser.add_argument('--id', type=str, default=114, help='The task type to test (e.g., "Action Classification")')
    parser.add_argument('--valid_file', type=str, 
                        default='/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge_hard.json',
                        help='Path to validation JSON file')
    parser.add_argument('--materials_folder', type=str,
                        default='/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge',
                        help='Materials folder path')
    parser.add_argument('--api_key', type=str, default="",
                        help='DeepSeek API key')
    
    args = parser.parse_args()

    # 设置环境变更
    os.environ['DEEPSEEK_API_KEY'] = args.api_key

    try:
        from multiagent_platform import EXECUTE_TOOL_CHAIN
    except ImportError as e:
        print(f"Error importing multiagent_platform: {e}")
        print("Please run this script from the SoccerAgent directory.")
        return

    print(f"Loading {args.valid_file}...")
    try:
        with open(args.valid_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {args.valid_file}")
        return

    # 寻找匹配的任务类型
    target_item = None
    for item in data:
        if item.get('id') == args.id:
            target_item = item
            break
    
    if not target_item:
        print(f"No id found with: '{args.id}'")
        print("Available ids (first 20 items):")
        seen_types = set()
        for item in data[:100]:
            t = item.get('id')
            if t and t not in seen_types:
                print(f" - {t}")
                seen_types.add(t)
        return

    print(f"ID: {target_item.get('id')}")
    print(f"Question: {target_item['Q']}")
    
    
    # 准备Prompt和Materials
    # question_id = target_item["id"]
    question = target_item["Q"]
    
    options_text = ""
    for i in range(1, 10):
        opt = target_item.get(f"O{i}")
        if opt:
            options_text += f"O{i}: {opt}\n"
    
    print("Options:")
    print(options_text)

    old_materials = target_item.get("materials", [])
    materials = []
    if old_materials:
        for m in old_materials:
            full_path = os.path.join(args.materials_folder, m)
            if os.path.exists(full_path):
                materials.append(full_path)
            else:
                print(f"Warning: Material not found at {full_path}")
    
    material_dict = {"materials": materials} if materials else None
    print(f"Materials: {materials}")

    print("\nRunning Agent...")
    try:
        result = EXECUTE_TOOL_CHAIN(
            query=question,
            material=material_dict,
            options=options_text
        )
        print("\nAgent Output:")
        print(result)
        
        answer = extract_option(result)
        print(f"\nExtracted Answer: {answer}")
        
        if "Answer" in target_item:
            print(f"Ground Truth: {target_item['Answer']}")
            if answer == target_item['Answer']:
                print("Result: CORRECT")
            else:
                print("Result: INCORRECT")
        
    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

import json
import os
import sys
import argparse
from tqdm import tqdm
from collections import defaultdict
import re
import random

os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"

# Add current directory to path so we can import multiagent_platform
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def extract_option(text):
    """Extract option answer from text"""
    pattern = r'\b(O[1-9])\b'
    # Find all matches
    matches = re.findall(pattern, text.upper())
    if matches:
        return matches[-1] # Return the last match
    return None

def evaluate_valid(input_file, materials_folder, api_key):
    # Set API key for SoccerAgent
    os.environ['DEEPSEEK_API_KEY'] = api_key

    try:
        from multiagent_platform import EXECUTE_TOOL_CHAIN
    except ImportError as e:
        print(f"Error importing multiagent_platform: {e}")
        print("Make sure you are running this script from the SoccerAgent directory.")
        return

    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit to first 200 questions
    print("Optimization: Evaluating only the first 200 questions.")
    data = data[:200]

    # Initialize stats
    total_processed = 0
    total_correct = 0
    
    # Task type stats: {type: {'total': 0, 'correct': 0}}
    task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    results_detail = []

    print(f"Starting evaluation on {len(data)} items...")
    
    for item in tqdm(data, desc="Evaluating"):
        question_id = item.get("id", "unknown")
        question = item.get("Q", "")
        closeA = item.get("closeA", "")
        task_type = item.get("task", "Unknown")
        
        # Prepare options text
        options_text = ""
        option_count = 0
        for i in range(1, 10):
            opt_key = f"O{i}"
            if opt_key in item and item[opt_key]:
                options_text += f"{opt_key}: {item[opt_key]}\n"
                option_count += 1
        
        # Prepare materials
        old_materials = item.get("materials", [])
        materials = []
        if old_materials:
            for m in old_materials:
                # Construct absolute path to material
                full_path = os.path.join(materials_folder, m)
                materials.append(full_path)
        
        material_dict = {"materials": materials} if materials else None

        # Run Agent
        predicted_option = None
        try:
            # Call the agent
            result = EXECUTE_TOOL_CHAIN(
                query=question,
                material=material_dict,
                options=options_text
            )
            predicted_option = extract_option(result)
        except Exception as e:
            print(f"\nError processing ID {question_id}: {e}")
            predicted_option = "Error"
        
        if not predicted_option:
            predicted_option = random.choice([f'O{i}' for i in range(1, option_count + 1)])

        # Check correctness
        is_correct = False
        if predicted_option and closeA:
            # Normalize to compare (e.g. "O1" vs "O1")
            if predicted_option.upper() == closeA.upper():
                is_correct = True
        
        # Update stats
        total_processed += 1
        if is_correct:
            total_correct += 1
        
        task_stats[task_type]['total'] += 1
        if is_correct:
            task_stats[task_type]['correct'] += 1
        
        # Print progress for each question
        print(f"ID: {question_id} | Pred: {predicted_option} | GT: {closeA} | {'CORRECT' if is_correct else 'WRONG'}")

            
        # Record detail
        results_detail.append({
            "id": question_id,
            "task": task_type,
            "question": question,
            "prediction": predicted_option,
            "ground_truth": closeA,
            "is_correct": is_correct
        })

    # Print Report
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    
    overall_acc = (total_correct / total_processed * 100) if total_processed > 0 else 0
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_processed})")
    
    print("\nAccuracy by Task Type:")
    print("-" * 80)
    print(f"{'Task Type':<40} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<10}")
    print("-" * 80)
    
    for t_type, stats in sorted(task_stats.items()):
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{t_type:<40} | {stats['total']:<6} | {stats['correct']:<8} | {acc:6.2f}%")
        
    print("-" * 80)

    # Save detailed results
    output_path = os.path.join(current_dir, "evaluation_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_detail, f, indent=4, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SoccerAgent on valid.json")
    parser.add_argument('--input_file', type=str, 
                        default='/root/autodl-tmp/SoccerNet_Challenge_VQA/test/test_with_tasks.json',
                        help='Path to valid.json')
    parser.add_argument('--materials_folder', type=str,
                        default='/root/autodl-tmp/SoccerNet_Challenge_VQA/test',
                        help='Materials folder path')
    parser.add_argument('--api_key', type=str, 
                        default="",
                        help='DeepSeek API Key')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        sys.exit(1)
        
    evaluate_valid(args.input_file, args.materials_folder, args.api_key)

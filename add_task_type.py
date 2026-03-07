import json
import argparse
import os

def add_task_type_field(input_file, output_file=None):
    if output_file is None:
        output_file = input_file

    print(f"Reading from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {input_file}.")
        return

    if not isinstance(data, list):
        print("Error: Root element of JSON is not a list.")
        return

    count = 0
    for item in data:
        if isinstance(item, dict):
            # 如果不存在 "task type" 字段，则添加
            if "task type" not in item:
                item["task type"] = ""
                count += 1
    
    print(f"Added 'task type' field to {count} items.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add 'task type': '' field to all items in a JSON list.")
    parser.add_argument('input_file', help="Path to the input JSON file")
    parser.add_argument('--output_file', help="Path to the output JSON file. Defaults to overwriting input file.", default=None)
    
    args = parser.parse_args()
    
    add_task_type_field(args.input_file, args.output_file)

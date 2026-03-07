import json
import os

def merge_results(base_file, hard_file, output_file):
    print(f"Loading base results from: {base_file}")
    with open(base_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    # Create a dictionary for quick lookup by ID
    # Assuming base_data is a list of dicts like [{"id": 1, "Answer": "O1"}, ...]
    base_dict = {item['id']: item for item in base_data}
    original_count = len(base_data)
    
    print(f"Loading hard results from: {hard_file}")
    with open(hard_file, 'r', encoding='utf-8') as f:
        hard_data = json.load(f)
        
    updated_count = 0
    new_count = 0
    
    for item in hard_data:
        question_id = item['id']
        answer = item['Answer']
        
        if question_id in base_dict:
            # Check if answer is actually different before counting as update? 
            # The user asked to "overwrite", so we just do it.
            if base_dict[question_id]['Answer'] != answer:
                base_dict[question_id]['Answer'] = answer
                updated_count += 1
        else:
            # If for some reason the ID doesn't exist in base, add it?
            # Usually we expect it to exist, but good to handle.
            base_dict[question_id] = item
            new_count += 1
            
    # Convert dict back to sorted list
    merged_list = sorted(base_dict.values(), key=lambda x: x['id'])
    
    print("-" * 40)
    print(f"Original items: {original_count}")
    print(f"Hard items to merge: {len(hard_data)}")
    print(f"Updated items: {updated_count}")
    print(f"New items added: {new_count}")
    print(f"Total items in output: {len(merged_list)}")
    print("-" * 40)
    
    print(f"Saving merged results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    base_path = "/root/autodl-tmp/SoccerAgent/agent_result_0222_merged_new_2.json"
    hard_path = "/root/autodl-tmp/SoccerAgent/agent_result_0222_new_camera.json"
    
    # Overwrite the base file directly, or create a new one?
    # User said "overwrite into agent_result_new.json", implying the file itself should be updated.
    # But for safety, maybe we write to a temp file first then rename, or just write directly since it's a script.
    # Let's write directly to agent_result_new.json as requested.
    
    output_path = "/root/autodl-tmp/SoccerAgent/agent_result_0222_merged_new_camera.json" 
    
    if not os.path.exists(base_path):
        print(f"Error: Base file not found: {base_path}")
        exit(1)
        
    if not os.path.exists(hard_path):
        print(f"Error: Hard file not found: {hard_path}")
        exit(1)
        
    merge_results(base_path, hard_path, output_path)

import json
import os

def prepare_datafiles(split_name, base_path, output_path):
    input_file = os.path.join(base_path, f"{split_name}.json")
    
    target_tasks = [
        "Game State Relevant QA",
        "Score and Time Relevant QA", 
        "Jersy Color Relevant QA",
        "Background knowledge Image QA",
        "Replay Grounding",
        "Foul",
        "Commentary Relevant QA"
    ]

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vlm_data = []
    
    for item in data:
        if item.get("task type") in target_tasks:
            materials = item.get("materials")
            if not materials:
                continue
            
            # Use the first material (usually one image/video)
            rel_path = materials[0]
            abs_path = os.path.join(base_path, rel_path)
            
            if not os.path.exists(abs_path):
                print(f"Warning: File not found {abs_path}")
                continue
                
            entry = {
                "id": len(vlm_data),
                "image": [abs_path], # Generic key for media
                "conversations": [
                    {
                        "role": "user",
                        "content": f"<image>\n{item['Q']}"
                    },
                    {
                        "role": "assistant",
                        "content": item['openA']
                    }
                ],
                "task_type": item.get("task type")
            }
            vlm_data.append(entry)

    print(f"Processed {len(vlm_data)} entries from {split_name}.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vlm_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_path}")

def prepare_data():
    # Process Train
    prepare_datafiles(
        "train",
        "/root/autodl-tmp/SoccerNet_Challenge_VQA/train",
        "/root/autodl-tmp/SoccerAgent/data_vlm_train.json"
    )
    
    # Process Valid
    prepare_datafiles(
        "valid",
        "/root/autodl-tmp/SoccerNet_Challenge_VQA/valid",
        "/root/autodl-tmp/SoccerAgent/data_vlm_valid.json"
    )

if __name__ == "__main__":
    prepare_data()

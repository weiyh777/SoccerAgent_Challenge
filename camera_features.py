import os
import json
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 配置路径
EXAMPLE_PATH = "/root/autodl-tmp/SoccerAgent/toolbox/utils/example_tiny"
OUTPUT_JSON = "/root/autodl-tmp/SoccerAgent/toolbox/utils/example_tiny/camera_features.json"
MODEL_PATH = "/root/autodl-tmp/SoccerAgent/model/Qwen2.5-VL-3B-Instruct"

# 13个标准类别
CAMERA_CATEGORIES = [
    "Close-up behind the goal", "Close-up corner", "Close-up player or field referee", 
    "Close-up side staff", "Goal line technology camera", "Inside the goal", 
    "Main behind the goal", "Main camera center", "Main camera left", 
    "Main camera right", "Other", "Public", "Spider camera"
]

def generate_camera_features():
    # 加载模型（建议量化加载以防提取时显存波动）
    print("Loading model for feature extraction...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 获取图片列表并排序，确保与类别对应
    example_files = sorted([f for f in os.listdir(EXAMPLE_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    feature_library = {}

    # 特征提取专用提示词
    extraction_prompt = extraction_prompt = """You are a professional football broadcast director. 
Analyze this specific camera angle and provide a DISCRIMINATIVE description to distinguish it from others.

Focus strictly on these discriminative features:
1. PITCH MARKINGS: Is the center circle, penalty area, goal line, or corner arc visible? 
2. PERSPECTIVE: Is the camera looking from the side, from behind the goal, or top-down? Are the pitch lines parallel or converging?
3. DEPTH OF FIELD: Is the background sharp (wide shot) or blurry (close-up)? 
4. CHARACTERISTIC OBJECTS: Can you see the goal net mesh (inside), the stadium roof (spider), or fan faces (public)?

Format: Describe the unique visual 'anchor' that defines this position. 
Example: "Main camera center is defined by the visibility of the center circle and perfectly horizontal side-lines from a high elevation."
"""

    for i in range(min(len(example_files), len(CAMERA_CATEGORIES))):
        img_path = os.path.join(EXAMPLE_PATH, example_files[i])
        category = CAMERA_CATEGORIES[i]
        
        print(f"Extracting features for: {category}...")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path, "max_pixels": 512*28*28}, # 限制像素减小开销
                    {"type": "text", "text": extraction_prompt}
                ]
            }
        ]

        # 推理逻辑
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        description = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        feature_library[category] = description.strip()

    # 保存为 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_library, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Feature library saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    generate_camera_features()
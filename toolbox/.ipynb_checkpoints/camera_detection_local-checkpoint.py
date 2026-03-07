import os
import json
import torch
import cv2
import base64
import re
from io import BytesIO
from PIL import Image
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 修改为真实路径
PROJECT_PATH = "/root/autodl-tmp/SoccerAgent" 
MODEL_PATH = "/root/autodl-tmp/SoccerAgent/model/Qwen2.5-VL-3B-Instruct"

# 全局变量缓存模型，避免重复加载
_model = None
_processor = None

def load_local_model():
    global _model, _processor
    if _model is None:
        print(f"Loading Camera Detection Model from {MODEL_PATH}...")
        try:
            adapter_config_path = os.path.join(MODEL_PATH, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, "r") as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path")
                
                print(f"Loading base model from {base_model_path}...")
                _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",
                    device_map="auto",
                )
                print(f"Loading LoRA adapter from {MODEL_PATH}...")
                _model.load_adapter(MODEL_PATH)
            else:
                _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa",  # 使用 sdpa 以适配 Blackwell/Ada 等新显卡
                    device_map="auto",
                )
            _processor = AutoProcessor.from_pretrained(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

def encode_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((256, 256))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_camera_position(reply):
    options = [
        "Main camera center", "Close-up player or field referee", "Close-up side staff", 
        "Main camera left", "Main behind the goal", "Close-up behind the goal", 
        "Spider camera", "Main camera right", "Public", "Goal line technology camera", 
        "Close-up corner", "Inside the goal", "Other"
    ]
    pattern = "|".join([re.escape(option) for option in options])
    match = re.search(pattern, reply, re.IGNORECASE)
    if match:
        # Return the canonical option name (capitalized correctly)
        for opt in options:
            if opt.lower() == match.group(0).lower():
                return opt
        return match.group(0)
    else:
        return "None"

def infer_with_qwen(prompt, img_base64=None, history_examples=[]):
    load_local_model()
    
    # 构建 Qwen 格式的 conversation
    messages = []
    
    # 1. 转换 Few-shot 样本 (History)
    # Qwen 不使用 'system' role 来传样本，我们直接作为前面的 user/assistant 对话，或者合并到当前的 user prompt 中
    # 为了简单起见，我们将 learning examples 转换为 message 历史
    
    # history_examples 结构是 OpenAI 格式: [{"role":..., "content": [...]}, ...]
    # 我们需要将其转换为 Qwen 可以理解的格式
    
    # 处理 "system" prompt (Qwen2.5-VL 支持 system，但通常放在这里)
    if history_examples and history_examples[0]['role'] == 'system':
        messages.append({"role": "system", "content": history_examples[0]['content']})
        start_idx = 1
    else:
        start_idx = 0
        
    for msg in history_examples[start_idx:]:
        qwen_msg = {"role": msg["role"], "content": []}
        if isinstance(msg["content"], list):
            for item in msg["content"]:
                if item["type"] == "text":
                    qwen_msg["content"].append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    # Qwen utils 也能识别 data URI
                    qwen_msg["content"].append({"type": "image", "image": item["image_url"]["url"]})
        else:
            qwen_msg["content"].append({"type": "text", "text": msg["content"]})
        messages.append(qwen_msg)

    # 2. 构建当前请求消息
    current_content = [{"type": "text", "text": prompt}]
    if img_base64:
        current_content.insert(0, {"type": "image", "image": f"data:image/png;base64,{img_base64}"})
    
    messages.append({"role": "user", "content": current_content})

    # 3. 推理
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_model.device)

    generated_ids = _model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = _processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def CAMERA_DETECTION(query=None, material=[]):
    if not material:
        return "No material provided for camera detection."

    example_path = f"{PROJECT_PATH}/toolbox/utils/example_tiny" # 修正路径
    
    # 构建 Few-shot 样本
    try:
        example_img = sorted([os.path.join(example_path, f) for f in os.listdir(example_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    except FileNotFoundError:
        print(f"Warning: Example path {example_path} not found. Running zero-shot.")
        example_img = []

    camera_position = [
        "Close-up behind the goal", "Close-up corner", "Close-up player or field referee", 
        "Close-up side staff", "Goal line technology camera", "Inside the goal", 
        "Main behind the goal", "Main camera center", "Main camera left", 
        "Main camera right", "Other", "Public", "Spider camera"
    ]

    learn_prompt = "I want you to help me identify the camera position of a football game photo. Now I will give you some example images, each of which corresponds to a specific camera position. Please learn the characteristics of these images for classification of new photos."

    history = [{"role": "system", "content": learn_prompt}]
    
    # 只有当样本图片数量和标签数量匹配时才添加样本
    limit = min(len(example_img), len(camera_position))
    for i in range(limit):
        base64_image = encode_image(example_img[i])
        # User 提供图片
        history.append({
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": "What is the camera position of this image?"}
            ]
        })
        # Assistant 回答确切位置 (模拟学习过程)
        history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": f"The camera position corresponding to this photo is: {camera_position[i]}"}]
        })

    img_path = material[0]
    file_extension = os.path.splitext(img_path)[1].lower()
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.webm']

    ask_prompt = "What is the camera position in this picture? The answer should be chosen from the following options: [Main camera center, Close-up player or field referee, Close-up side staff, Main camera left, Main behind the goal, Close-up behind the goal, Spider camera, Main camera right, Public, Goal line technology camera, Close-up corner, Inside the goal, Other]. Please answer directly with the option name."

    if file_extension in image_extensions:
        base64_image = encode_image(img_path)
        reply = infer_with_qwen(ask_prompt, base64_image, history)
        ans = extract_camera_position(reply)
        return f"The camera position in the photo is: {ans}."
    
    elif file_extension in video_extensions:
        # 处理视频：采样帧
        video_path = img_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"Cannot open video: {video_path}"
        
        frames_base64 = []
        frame_count = 0
        max_frames = 5 # 限制处理帧数，提高速度
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀采样 5 帧
        interval = max(1, total_frames // max_frames)
        
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame % interval == 0 and len(frames_base64) < max_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                # Resize if too large to speed up
                pil_image.thumbnail((720, 720)) 
                img_byte_arr = BytesIO()
                pil_image.save(img_byte_arr, format='JPEG', quality=85) # 使用 JPEG 压缩
                img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                frames_base64.append(img_base64)
            current_frame += 1
            
        cap.release()
        
        reply = []
        for i, frame in enumerate(frames_base64):
            # 视频帧检测时，可以简化 Prompt，不一定每次都带完整的 history，根据显存情况决定
            # 这里为了效果仍带上 history
            result = infer_with_qwen(ask_prompt, frame, history)
            reply.append(result)
            
        ans_list = [extract_camera_position(r) for r in reply]
        # 过滤掉 None
        ans_list = [a for a in ans_list if a != "None"]
        if not ans_list:
            return "The camera position in the video is: None."
            
        count = Counter(ans_list)
        most_common_str, most_common_count = count.most_common(1)[0]
        return f"The camera position in the video is: {most_common_str}."
        
    return "Unsupported file format."

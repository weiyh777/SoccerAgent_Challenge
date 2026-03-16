# SoccerNet VQA Challenge 2026 - 技术路线指南

## 目录
1. [快速开始：获取Challenge提交结果](#1-快速开始获取challenge提交结果)
2. [SoccerAgent架构深入理解](#2-socceragent架构深入理解)
3. [微调技术路线](#3-微调技术路线)
4. [高级优化策略](#4-高级优化策略)
5. [ReAct架构详解与SoccerAgent分析](ReAct_Architecture.md)

---

## 1. 快速开始：获取Challenge提交结果

### 1.1 方法一：使用Baseline (VLM直接推理)

最简单的方式是使用提供的baseline模型直接生成预测结果：

```bash
# 使用 Qwen2.5-VL 模型
cd /root/autodl-tmp/SoccerAgent/baseline_challenge

python baseline.py \
    --model qwen \
    --input_file /root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge.json \
    --output_file /root/autodl-tmp/SoccerAgent/baseline_challenge/metadata.json \
    --materials_folder /root/autodl-tmp/SoccerNet_Challenge_VQA/challenge \
    --model_path <YOUR_QWEN2.5VL_MODEL_PATH>

# 或使用 GPT-4o 模型
python baseline.py \
    --model gpt \
    --input_file /root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge.json \
    --output_file /root/autodl-tmp/SoccerAgent/baseline_challenge/metadata.json \
    --materials_folder /root/autodl-tmp/SoccerNet_Challenge_VQA/challenge \
    --api_key <YOUR_OPENAI_API_KEY>
```

### 1.2 方法二：使用SoccerAgent多智能体系统

SoccerAgent提供了更强大的多工具链推理能力：

```python
# run_challenge.py
import json
import os
import sys
sys.path.append('/root/autodl-tmp/SoccerAgent')

from multiagent_platform import EXECUTE_TOOL_CHAIN

def run_challenge(input_file, output_file, materials_folder):
    """
    使用SoccerAgent工具链执行Challenge
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    for item in data:
        question_id = item["id"]
        question = item["Q"]
        
        # 构建选项字符串
        options = ""
        for i in range(1, 10):
            opt = item.get(f"O{i}")
            if opt:
                options += f"O{i}: {opt}\n"
        
        # 处理材料路径
        materials = item.get("materials", [])
        if materials:
            materials = [os.path.join(materials_folder, m) for m in materials]
        
        try:
            # 调用SoccerAgent工具链
            result = EXECUTE_TOOL_CHAIN(
                query=question,
                material={"materials": materials} if materials else None,
                options=options
            )
            
            # 从结果中提取答案 (O1, O2, O3, O4...)
            answer = extract_answer_from_result(result)
        except Exception as e:
            print(f"Error on question {question_id}: {e}")
            answer = "O1"  # 默认答案
        
        results.append({"id": question_id, "Answer": answer})
        print(f"Processed question {question_id}: {answer}")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")

def extract_answer_from_result(result):
    """从Agent结果中提取选项答案"""
    import re
    # 尝试匹配 O1, O2, O3, O4 等格式
    pattern = r'\b(O[1-9])\b'
    matches = re.findall(pattern, result)
    if matches:
        return matches[-1]  # 返回最后一个匹配的选项
    return "O1"

if __name__ == "__main__":
    run_challenge(
        input_file="/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge.json",
        output_file="/root/autodl-tmp/SoccerAgent/baseline_challenge/metadata.json",
        materials_folder="/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge"
    )
```

### 1.3 生成提交文件

```bash
# 生成提交用的zip文件
cd /root/autodl-tmp/SoccerAgent/baseline_challenge
zip submission.zip metadata.json
```

提交格式要求：
```json
[
    {"id": 1, "Answer": "O3"},
    {"id": 2, "Answer": "O4"},
    {"id": 3, "Answer": "O1"},
    ...
]
```

---

## 2. SoccerAgent架构深入理解

### 2.1 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     SoccerAgent Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 1: Task Decomposition                                     │
│  ├── 问题分析 → 识别已知信息 (Known Info)                          │
│  └── 工具链规划 → 生成 Tool Chain                                  │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 2: Tool Execution                                         │
│  ├── 依次执行工具链中的每个工具                                     │
│  ├── 收集中间结果                                                  │
│  └── 直到 <EndCall> 标记                                          │
├─────────────────────────────────────────────────────────────────┤
│  PHASE 3: Solution Synthesis                                     │
│  └── LLM 综合所有结果生成最终答案                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 可用工具列表 (18个工具)

| 工具名称 | 功能 | 适用任务 |
|---------|------|---------|
| `Textual Entity Search` | 搜索球员/球队/裁判的Wiki信息 | 背景知识QA |
| `Textual Retrieval Augment` | 从数据库检索相关信息 | 背景知识QA |
| `Game Search` | 搜索比赛信息 | 比赛情况QA |
| `Game Info Retrieval` | 获取比赛基本信息 | 比赛情况QA |
| `Match History Retrieval` | 获取比赛历史事件 | 比赛事件统计QA |
| `Camera Detection` | 检测相机位置类型 | 相机状态分类 |
| `Shot Change` | 检测视频中的镜头切换 | 相机切换分类 |
| `Number Recognition` | 识别球衣号码 | 球衣号码QA |
| `Entity Recognition` | 人脸识别球员 | 背景知识图像QA |
| `Score and Time Recognition` | 识别比分和时间 | 比分时间QA |
| `Vision Language Model` | 通用视觉问答 | 多种任务 |
| `Jersey Color Relevant VQA` | 球衣颜色相关问答 | 球衣颜色QA |
| `Action Classifier` | 动作分类 (24类) | 动作分类 |
| `Commentary Generation` | 生成解说文本 | 解说生成 |
| `Replay Grounding` | 回放视频定位 | 回放定位 |
| `Foul Recognition` | 犯规识别 | 多视角犯规识别 |
| `Frame Selection` | 从视频选取关键帧 | 辅助工具 |
| `Segment` | 图像分割 | 辅助工具 |

### 2.3 任务类型与推荐工具链

根据 `tasks.csv`，14种任务类型的推荐工具链：

| 任务类型 | 推荐工具链 |
|---------|-----------|
| Background knowledge text QA | `Textual Entity Search → Textual Retrieval Augment → LLM` |
| Match Situation QA | `Game Search → Game Info Retrieval → Match History Retrieval → LLM` |
| Camera Status Classification | `Camera Detection → LLM` |
| Background knowledge Image QA | `Entity Recognition → Textual Entity Search → Textual Retrieval Augment → LLM` |
| Jersey Number Recognition | `Number Recognition → LLM` |
| Score and Time Relevant QA | `Score and Time Recognition → LLM` |
| Game State Relevant QA | `Vision Language Model → LLM` |
| Camera Status Switching | `Shot Change → Camera Detection (twice) → LLM` |
| Replay Grounding | `Replay Grounding → LLM` 或 `Commentary Generation (5x) → LLM` |
| Action Classification | `Action Classifier → LLM` |
| Commentary Generation | `Commentary Generation → LLM` |
| Commentary Relevant QA | `Vision Language Model → LLM` 或 `Frame Selection → Entity Recognition → Textual Retrieval Augment → LLM` |
| Jersey Color Relevant QA | `Vision Language Model → LLM` 或 `Segment → Vision Language Model → LLM` |
| Multi-view Foul Recognition | `Foul Recognition → LLM` |

---

## 3. 微调技术路线

### 3.1 路线一：微调VLM基座模型

**目标**: 提升VLM在足球领域的理解能力

#### 3.1.1 数据准备

```python
# prepare_finetune_data.py
import json
import os

def prepare_vlm_finetune_data(train_json, materials_folder, output_file):
    """
    将训练数据转换为VLM微调格式
    """
    with open(train_json, 'r') as f:
        data = json.load(f)
    
    finetune_data = []
    for item in data:
        question = item["Q"]
        answer = item["closeA"]  # 正确选项
        open_answer = item.get("openA", "")  # 开放式答案
        task_type = item.get("task type", "")
        
        # 构建选项文本
        options_text = ""
        correct_option_text = ""
        for i in range(1, 10):
            opt = item.get(f"O{i}")
            if opt:
                options_text += f"O{i}: {opt}\n"
                if f"O{i}" == answer:
                    correct_option_text = opt
        
        # 处理材料
        materials = item.get("materials", [])
        if materials:
            materials = [os.path.join(materials_folder, m) for m in materials]
        
        # 构建微调样本
        sample = {
            "conversations": [
                {
                    "role": "user",
                    "content": f"Q: {question}\n{options_text}"
                },
                {
                    "role": "assistant", 
                    "content": f"{answer}"  # 或者 f"{answer}: {correct_option_text}"
                }
            ],
            "images": materials if materials else [],
            "task_type": task_type
        }
        finetune_data.append(sample)
    
    with open(output_file, 'w') as f:
        json.dump(finetune_data, f, indent=2)
    
    print(f"Prepared {len(finetune_data)} samples for finetuning")

# 执行
prepare_vlm_finetune_data(
    train_json="/root/autodl-tmp/SoccerNet_Challenge_VQA/train/train.json",
    materials_folder="/root/autodl-tmp/SoccerNet_Challenge_VQA/train",
    output_file="/root/autodl-tmp/SoccerAgent/finetune_data/train_vlm.json"
)
```

#### 3.1.2 Qwen2.5-VL 微调脚本

```python
# finetune_qwen_vl.py
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json

def load_finetune_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def main():
    # 加载模型
    model_path = "Qwen/Qwen2.5-VL-7B-Instruct"  # 或更大的模型
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    # LoRA配置 (参数高效微调)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./soccer_vlm_lora",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )
    
    # 加载数据
    train_dataset = load_finetune_data("./finetune_data/train_vlm.json")
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    trainer.train()
    model.save_pretrained("./soccer_vlm_lora_final")

if __name__ == "__main__":
    main()
```

### 3.2 路线二：优化多智能体系统

**目标**: 提升任务分解和工具链执行的准确性

#### 3.2.1 改进Task Decomposition Prompt

```python
# improved_task_decomposition.py

IMPROVED_TASK_DECOMPOSITION_PROMPT = """
# Soccer Question Answering Assistant - Enhanced Version

## Task Classification
Based on the question content and materials, first classify the question into one of these 14 categories:

1. **Background knowledge text QA** - Questions about players, teams, referees, venues (no image/video)
2. **Match Situation QA** - Questions about specific match events/statistics
3. **Match Events and Statistical QA** - Questions about match history events
4. **Camera Status Classification** - Identify camera position in images
5. **Background knowledge Image QA** - Questions about entities with images provided
6. **Jersey Number Recognition** - Detect player jersey numbers in images
7. **Score and Time Relevant QA** - Extract score/time from broadcast images
8. **Game State Relevant QA** - Count players, analyze formations in images
9. **Camera Status Switching** - Classify camera transitions in videos
10. **Replay Grounding** - Identify which action is being replayed
11. **Action Classification** - Recognize soccer actions in videos
12. **Commentary Generation** - Generate textual descriptions from videos
13. **Commentary Relevant QA** - Questions combining player info and commentary
14. **Jersey Color Relevant QA** - Questions involving jersey color information
15. **Multi-view Foul Recognition** - Analyze fouls from multiple angles

## Decision Rules
- If question mentions "jersey number" or "shirt number" → Jersey Number Recognition
- If question asks about camera/viewpoint → Camera Status Classification/Switching
- If question asks "describe/generate commentary" → Commentary Generation
- If question mentions "score" or "time" with broadcast image → Score and Time Relevant QA
- If question about player/team history without images → Background knowledge text QA
- If question about match with specific date/teams → Match Situation QA
- If video shows foul with multiple angles → Multi-view Foul Recognition
- If question asks about "replay" with multiple videos → Replay Grounding
- If question asks about action type in video → Action Classification
- If question involves jersey colors → Jersey Color Relevant QA
- If question asks about player count/formation → Game State Relevant QA

## Material Analysis
- **No materials**: Likely text-based QA (Background knowledge, Match Situation)
- **Single image**: Camera Detection, Jersey Number, Score/Time, Entity Recognition, Game State
- **Multiple images**: Jersey Number (from different angles)
- **Single video**: Action Classification, Commentary Generation, Camera Switching
- **Multiple videos**: Replay Grounding, Multi-view Foul Recognition

## Your Response Format
Task Type: [Identified task type]
Known Info: [$category1$, $category2$, ...]
Tool Chain: [*Tool1* -> *Tool2* -> ... -> *LLM*]
"""
```

#### 3.2.2 添加任务类型检测器

```python
# task_classifier.py
import re

def classify_question(question, materials):
    """
    基于规则的任务类型分类器
    """
    q_lower = question.lower()
    
    # 材料类型分析
    has_image = False
    has_video = False
    multi_video = False
    
    if materials:
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
        
        for m in materials:
            ext = m.lower().split('.')[-1]
            if f'.{ext}' in image_exts:
                has_image = True
            elif f'.{ext}' in video_exts:
                has_video = True
        
        if has_video and len(materials) > 1:
            multi_video = True
    
    # 规则匹配
    if "jersey number" in q_lower or "shirt number" in q_lower:
        return "Jersey Number Recognition", "*Number Recognition* -> *LLM*"
    
    if "camera" in q_lower and "switch" in q_lower:
        return "Camera Status Switching", "*Shot Change* -> *Camera Detection* -> *LLM*"
    
    if "camera" in q_lower or "viewpoint" in q_lower:
        return "Camera Status Classification", "*Camera Detection* -> *LLM*"
    
    if "score" in q_lower and has_image:
        return "Score and Time Relevant QA", "*Score and Time Recognition* -> *LLM*"
    
    if "time" in q_lower and ("half" in q_lower or "minute" in q_lower) and has_image:
        return "Score and Time Relevant QA", "*Score and Time Recognition* -> *LLM*"
    
    if "commentary" in q_lower or "describe" in q_lower:
        return "Commentary Generation", "*Commentary Generation* -> *LLM*"
    
    if "category" in q_lower and "video" in q_lower:
        return "Action Classification", "*Action Classifier* -> *LLM*"
    
    if "replay" in q_lower and multi_video:
        return "Replay Grounding", "*Replay Grounding* -> *LLM*"
    
    if "foul" in q_lower and multi_video:
        return "Multi-view Foul Recognition", "*Foul Recognition* -> *LLM*"
    
    if "jersey color" in q_lower or "shirt color" in q_lower:
        return "Jersey Color Relevant QA", "*Jersey Color Relevant VQA* -> *LLM*"
    
    if "count" in q_lower or "how many players" in q_lower:
        return "Game State Relevant QA", "*Vision Language Model* -> *LLM*"
    
    if has_image and not has_video:
        # 可能是背景知识图像QA
        return "Background knowledge Image QA", "*Entity Recognition* -> *Textual Entity Search* -> *Textual Retrieval Augment* -> *LLM*"
    
    if has_video and ("who" in q_lower or "player" in q_lower):
        return "Commentary Relevant QA", "*Vision Language Model* -> *LLM*"
    
    # 文本类问题
    if not materials:
        if re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', question) or "match" in q_lower:
            return "Match Situation QA", "*Game Search* -> *Game Info Retrieval* -> *Match History Retrieval* -> *LLM*"
        return "Background knowledge text QA", "*Textual Entity Search* -> *Textual Retrieval Augment* -> *LLM*"
    
    # 默认
    return "Unknown", "*Vision Language Model* -> *LLM*"
```

### 3.3 路线三：混合方法 (推荐)

结合VLM微调和多智能体优化，形成最佳实践：

```python
# hybrid_soccer_agent.py
import json
import os
from task_classifier import classify_question

class HybridSoccerAgent:
    def __init__(self, 
                 vlm_model_path,
                 deepseek_api_key,
                 database_path="/root/autodl-tmp/SoccerAgent/database"):
        """
        混合足球问答Agent
        - 使用微调后的VLM处理视觉相关任务
        - 使用多智能体工具链处理知识检索任务
        """
        self.vlm = self._load_vlm(vlm_model_path)
        self.api_key = deepseek_api_key
        self.database_path = database_path
        
        # 加载工具
        from toolbox import (
            GAME_SEARCH, TEXTUAL_ENTITY_SEARCH, TEXTUAL_RETRIEVAL_AUGMENT,
            MATCH_HISTORY_RETRIEVAL, GAME_INFO_RETRIEVAL, CAMERA_DETECTION,
            JERSEY_NUMBER_RECOGNITION, SCORE_TIME_DETECTION, ACTION_CLASSIFICATION,
            COMMENTARY_GENERATION, REPLAY_GROUNDING, FOUL_RECOGNITION
        )
        
        self.tools = {
            "Game Search": GAME_SEARCH,
            "Textual Entity Search": TEXTUAL_ENTITY_SEARCH,
            "Textual Retrieval Augment": TEXTUAL_RETRIEVAL_AUGMENT,
            "Match History Retrieval": MATCH_HISTORY_RETRIEVAL,
            "Game Info Retrieval": GAME_INFO_RETRIEVAL,
            "Camera Detection": CAMERA_DETECTION,
            "Number Recognition": JERSEY_NUMBER_RECOGNITION,
            "Score and Time Recognition": SCORE_TIME_DETECTION,
            "Action Classifier": ACTION_CLASSIFICATION,
            "Commentary Generation": COMMENTARY_GENERATION,
            "Replay Grounding": REPLAY_GROUNDING,
            "Foul Recognition": FOUL_RECOGNITION,
        }
    
    def _load_vlm(self, model_path):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_path)
        return {"model": model, "processor": processor}
    
    def answer(self, question, options, materials=None):
        """
        回答问题的主入口
        """
        # Step 1: 分类任务类型
        task_type, tool_chain = classify_question(question, materials)
        
        # Step 2: 根据任务类型选择处理策略
        if task_type in ["Camera Status Classification", "Jersey Number Recognition", 
                         "Score and Time Relevant QA", "Game State Relevant QA",
                         "Camera Status Switching", "Action Classification",
                         "Commentary Generation"]:
            # 视觉密集型任务 - 直接使用VLM
            return self._vlm_direct_answer(question, options, materials)
        
        elif task_type in ["Background knowledge text QA", "Match Situation QA"]:
            # 知识检索任务 - 使用工具链
            return self._tool_chain_answer(question, options, materials, tool_chain)
        
        elif task_type in ["Background knowledge Image QA", "Commentary Relevant QA", 
                           "Jersey Color Relevant QA"]:
            # 混合任务 - 先VLM提取信息，再知识检索
            return self._hybrid_answer(question, options, materials, tool_chain)
        
        else:
            # 默认使用VLM
            return self._vlm_direct_answer(question, options, materials)
    
    def _vlm_direct_answer(self, question, options, materials):
        """使用VLM直接回答"""
        prompt = f"Q: {question}\n{options}\nPlease answer with O1, O2, O3, or O4."
        
        # 构建对话
        from qwen_vl_utils import process_vision_info
        
        conversation = [
            {"role": "system", "content": "You are a football expert. Answer with only the option number."},
            {"role": "user", "content": []}
        ]
        
        # 添加图像/视频
        if materials:
            for m in materials:
                if m.endswith(('.jpg', '.png', '.jpeg')):
                    conversation[1]["content"].append({"type": "image", "image": f"file://{m}"})
                elif m.endswith(('.mp4', '.mkv', '.avi')):
                    conversation[1]["content"].append({
                        "type": "video", "video": f"file://{m}",
                        "max_pixels": 360 * 640, "fps": 1.0
                    })
        
        conversation[1]["content"].append({"type": "text", "text": prompt})
        
        # 推理
        text = self.vlm["processor"].apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.vlm["processor"](
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.vlm["model"].device)
        
        output_ids = self.vlm["model"].generate(**inputs, max_new_tokens=16)
        response = self.vlm["processor"].batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]
        
        return self._extract_option(response)
    
    def _tool_chain_answer(self, question, options, materials, tool_chain):
        """使用工具链回答"""
        # 解析工具链
        tools = [t.strip().replace('*', '') for t in tool_chain.split('->') if t.strip() != 'LLM']
        
        context = ""
        for tool_name in tools:
            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](question, materials)
                    context += f"\n[{tool_name} Result]: {result}"
                except Exception as e:
                    context += f"\n[{tool_name} Error]: {str(e)}"
        
        # 使用LLM综合答案
        final_prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}
Options:
{options}

Answer with only O1, O2, O3, or O4."""
        
        return self._llm_answer(final_prompt)
    
    def _hybrid_answer(self, question, options, materials, tool_chain):
        """混合方法"""
        # 先用VLM提取视觉信息
        visual_info = self._vlm_extract_info(question, materials)
        
        # 再用工具链检索知识
        enhanced_question = f"{question}\nVisual context: {visual_info}"
        return self._tool_chain_answer(enhanced_question, options, materials, tool_chain)
    
    def _vlm_extract_info(self, question, materials):
        """使用VLM提取视觉信息"""
        prompt = f"Describe what you see in this image/video that is relevant to: {question}"
        # ... VLM推理代码 ...
        return "extracted visual information"
    
    def _llm_answer(self, prompt):
        """使用LLM回答"""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        return self._extract_option(response.choices[0].message.content)
    
    def _extract_option(self, text):
        """从文本中提取选项"""
        import re
        match = re.search(r'\b(O[1-4])\b', text)
        return match.group(1) if match else "O1"
```

---

## 4. 高级优化策略

### 4.1 数据增强

```python
# data_augmentation.py
import random

def augment_qa_data(qa_item):
    """
    数据增强策略
    """
    augmented = []
    
    # 1. 选项顺序打乱
    options = [(f"O{i}", qa_item.get(f"O{i}")) for i in range(1, 5) if qa_item.get(f"O{i}")]
    random.shuffle(options)
    
    new_item = qa_item.copy()
    new_correct = None
    for i, (old_key, value) in enumerate(options, 1):
        new_item[f"O{i}"] = value
        if old_key == qa_item["closeA"]:
            new_correct = f"O{i}"
    new_item["closeA"] = new_correct
    augmented.append(new_item)
    
    # 2. 问题改写 (可使用LLM)
    # ...
    
    return augmented
```

### 4.2 模型集成 (Ensemble)

```python
# ensemble_agent.py

class EnsembleAgent:
    def __init__(self, agents):
        self.agents = agents
    
    def answer(self, question, options, materials):
        votes = {}
        for agent in self.agents:
            ans = agent.answer(question, options, materials)
            votes[ans] = votes.get(ans, 0) + 1
        
        # 多数投票
        return max(votes.keys(), key=lambda x: votes[x])
```

### 4.3 错误分析与针对性优化

```python
# error_analysis.py

def analyze_errors(predictions, ground_truth, data):
    """
    分析错误类型，找出薄弱环节
    """
    errors_by_task = {}
    
    for pred, gt, item in zip(predictions, ground_truth, data):
        if pred != gt:
            task_type = item.get("task type", "Unknown")
            if task_type not in errors_by_task:
                errors_by_task[task_type] = []
            errors_by_task[task_type].append({
                "question": item["Q"],
                "predicted": pred,
                "correct": gt
            })
    
    # 按错误数量排序
    sorted_errors = sorted(errors_by_task.items(), key=lambda x: len(x[1]), reverse=True)
    
    print("Error Analysis by Task Type:")
    for task, errors in sorted_errors:
        print(f"\n{task}: {len(errors)} errors")
        for e in errors[:3]:  # 显示前3个错误
            print(f"  Q: {e['question'][:50]}...")
            print(f"  Pred: {e['predicted']}, Correct: {e['correct']}")
    
    return errors_by_task
```

### 4.4 推荐的完整流程

```bash
# 1. 环境准备
conda create -n soccer_vqa python=3.10
conda activate soccer_vqa
pip install -r requirements.txt

# 2. 数据预处理
python prepare_finetune_data.py

# 3. 微调VLM (可选，耗时较长)
python finetune_qwen_vl.py

# 4. 在验证集上测试
python evaluate.py --input valid.json --model_path ./soccer_vlm_lora_final

# 5. 生成Challenge预测
python run_challenge.py --input challenge.json --output metadata.json

# 6. 打包提交
zip submission.zip metadata.json
# 上传到 Codabench: https://www.codabench.org/competitions/11087/
```

---

## 附录：关键路径说明

| 路径 | 说明 |
|-----|------|
| `/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/challenge.json` | Challenge测试数据 |
| `/root/autodl-tmp/SoccerNet_Challenge_VQA/challenge/materials/` | Challenge图像/视频素材 |
| `/root/autodl-tmp/SoccerNet_Challenge_VQA/train/train.json` | 训练数据 (~10k样本) |
| `/root/autodl-tmp/SoccerNet_Challenge_VQA/valid/valid.json` | 验证数据 |
| `/root/autodl-tmp/SoccerAgent/multiagent_platform.py` | 多智能体核心代码 |
| `/root/autodl-tmp/SoccerAgent/toolbox/` | 18个工具实现 |
| `/root/autodl-tmp/SoccerAgent/baseline_challenge/` | Challenge专用baseline |

---

**祝你在SoccerNet Challenge 2026中取得好成绩！🏆⚽**

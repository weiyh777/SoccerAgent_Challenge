from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import sys
sys.path.append('/root/autodl-tmp/SoccerAgent/toolbox')
# from utils.all_devices import vlm_device
from utils.all_devices import vlm_device

DEVICE = vlm_device
print("VLM on:", DEVICE)

# default: Load the model on the available device(s)
vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/SoccerAgent/model/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map=DEVICE
)
vlm_model.eval()

# default processer
vlm_processor = AutoProcessor.from_pretrained("/root/autodl-tmp/SoccerAgent/model/Qwen2.5-VL-3B-Instruct")

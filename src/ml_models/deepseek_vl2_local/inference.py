from typing import Optional
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor
from PIL import Image

# repo root から相対パスで指定
ROOT = Path(__file__).parents[3]
DEFAULT_MODEL_DIR = (ROOT / "models" / "deepseek-vl2-small").resolve()

class DeepSeekVL2Local:
    def __init__(self, model_dir: str = None, device: str = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        print("ising device is ", self.device)

    def load(self, dtype=torch.float16):
        if self.model is not None:
            return

        print(f"[DeepSeekVL2Local] loading model from {self.model_dir}")
        self.processor = DeepseekVLV2Processor.from_pretrained(str(self.model_dir))

        torch_dtype = dtype if self.device.startswith("cuda") else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        ) # .to(self.device) device_map="auto"では不要
        self.model.eval()
    
    def analyze_image(self, image_path: Optional[str], prompt: str, max_new_tokens: int = 256):
        if self.model is None:
            self.load()

        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        else:
            image = None
        # プロンプトと画像をモデル入力形式に変換
        conversation = [{"role": "User", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            text = self.processor.batch_decode(output, skip_special_tokens=True)[0]

        return text

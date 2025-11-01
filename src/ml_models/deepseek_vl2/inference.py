from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# モデルがリポジトリ直下の /models/deepseek-vl2 にある想定
ROOT = Path(__file__).parents[4]  # src/ml_models/deepseek_vl2/ -> repo root
DEFAULT_MODEL_DIR = ROOT / "models" / "deepseek-vl2"

class DeepSeekVL2Local:
    def __init__(self, model_dir: str = None, device: str = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ロードは重いので遅延ロードにする
        self.processor = None
        self.model = None

    def load(self, dtype=torch.float16):
        if self.processor is None:
            print(f"[DeepSeekVL2Local] loading from {self.model_dir} to {self.device}")
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir))
            torch_dtype = dtype if self.device.startswith("cuda") else torch.float32
            self.model = AutoModelForVision2Seq.from_pretrained(
                str(self.model_dir),
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()

    def analyze_image(self, image_path: str, prompt: str, max_new_tokens: int = 256) -> str:
        if self.model is None:
            self.load()
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text

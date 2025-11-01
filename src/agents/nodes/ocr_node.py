# 必要ならOCRでUIテキストを抽出するノード（DeepSeek-OCR利用想定）
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from pathlib import Path

class OCRInput(BaseModel):
    image_path: str

class OCROutput(BaseModel):
    text: str

# 簡易実装（もし DeepSeek-OCR を models に置いているなら同様のやり方でロード）
OCR_MODEL_DIR = Path.cwd() / "models" / "deepseek-ocr"

_processor = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def run_ocr(payload: OCRInput) -> OCROutput:
    global _processor, _model
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(str(OCR_MODEL_DIR))
        _model = AutoModelForVision2Seq.from_pretrained(str(OCR_MODEL_DIR)).to(_device)
        _model.eval()
    image = Image.open(payload.image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        ids = _model.generate(**inputs)
        text = _processor.batch_decode(ids, skip_special_tokens=True)[0]
    return OCROutput(text=text)

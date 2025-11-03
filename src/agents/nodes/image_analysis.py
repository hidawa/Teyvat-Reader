from pydantic import BaseModel
from typing import Optional
from src.ml_models.deepseek_vl2_local.inference import DeepSeekVL2Local

# シンプルな Node 入出力用 Pydantic を内部で作る（LangGraph連携の型に合わせやすく）
class ImageAnalysisInput(BaseModel):
    image_path: Optional[str] = None
    prompt: Optional[str]

class ImageAnalysisOutput(BaseModel):
    summary: str

# 単一インスタンス（重いロードを避ける）
_vl2 = DeepSeekVL2Local()

def run_image_analysis(payload: ImageAnalysisInput) -> ImageAnalysisOutput:
    prompt = payload.prompt
    img = payload.image_path
    res = _vl2.analyze_image(img, prompt)
    return ImageAnalysisOutput(summary=res)

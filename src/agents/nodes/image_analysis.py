from pydantic import BaseModel
from src.ml_models.deepseek_vl2.inference import DeepSeekVL2Local

# シンプルな Node 入出力用 Pydantic を内部で作る（LangGraph連携の型に合わせやすく）
class ImageAnalysisInput(BaseModel):
    image_path: str
    prompt: str

class ImageAnalysisOutput(BaseModel):
    summary: str

# 単一インスタンス（重いロードを避ける）
_vl2 = DeepSeekVL2Local()

def run_image_analysis(payload: ImageAnalysisInput) -> ImageAnalysisOutput:
    prompt = payload.prompt
    img = payload.image_path
    res = _vl2.analyze_image(img, prompt)
    return ImageAnalysisOutput(summary=res)

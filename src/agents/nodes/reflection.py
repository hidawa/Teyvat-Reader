from pydantic import BaseModel
from typing import List
import time

from src.ml_models.deepseek_vl2_local.inference import DeepSeekVL2Local

class ReflectionInput(BaseModel):
    initial_explanation: str
    feedbacks: List[str]

class ReflectionOutput(BaseModel):
    refined: str

def run_reflection(payload: ReflectionInput) -> ReflectionOutput:
    # シンプルなリフレクションの実装例：
    # hits（KB検索結果）を取り込んで解釈を洗練する
    base = payload.initial_explanation
    _vl2 = DeepSeekVL2Local()

    res = _vl2.analyze_image(prompt=base, image_path=None)
    
    # 少し時間を置く（本来は LLM で再生成）
    time.sleep(0.1)
    return ReflectionOutput(refined=res)

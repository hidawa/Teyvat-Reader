from pydantic import BaseModel
from typing import List
import time

class ReflectionInput(BaseModel):
    initial_explanation: str
    hits: List[dict] = []

class ReflectionOutput(BaseModel):
    refined: str

def run_reflection(payload: ReflectionInput) -> ReflectionOutput:
    # シンプルなリフレクションの実装例：
    # hits（KB検索結果）を取り込んで解釈を洗練する
    base = payload.initial_explanation
    hits = payload.hits or []
    refined = base
    # 簡易ルール：ヒットがあれば要素を付け加える
    if hits:
        additions = []
        for h in hits[:3]:
            name = h.get("name") or h.get("title") or "関連項目"
            additions.append(f"{name}に関連する可能性があります")
        refined = base + "\n追加情報: " + "; ".join(additions)
    # 少し時間を置く（本来は LLM で再生成）
    time.sleep(0.1)
    return ReflectionOutput(refined=refined)

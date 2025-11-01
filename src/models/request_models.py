from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    text: str
    # image_path はアップロード後のローカルパス（Dash が保存）
    image_path: Optional[str] = None
    # reflection 回数（任意）
    reflection_rounds: int = 2
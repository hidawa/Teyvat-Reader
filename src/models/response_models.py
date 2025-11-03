from pydantic import BaseModel
from typing import List, Optional

# class NodeOutput(BaseModel):
#     node: str
#     content: str

class AgentResponse(BaseModel):
    final_answer: str
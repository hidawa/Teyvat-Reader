"""
LangGraph 風にノードを定義して動かすラッパー。
ここでは依存を少なくするために簡易オーケストレータ実装を使っています。
（本番は langgraph を使う場合は Graph を置き換えてください）
"""
from typing import Any, Dict
from pydantic import BaseModel
from .nodes.image_analysis import ImageAnalysisInput, run_image_analysis
from .nodes.knowledge_search import KnowledgeSearchInput, run_knowledge_search
from .nodes.reflection import ReflectionInput, run_reflection
from .nodes.ocr_node import OCRInput, run_ocr

from src.models.response_models import AgentResponse, NodeOutput

import asyncio

async def run_agent(chat_request):
    """
    chat_request is instance of src.models.request_models.ChatRequest or dict-like
    Flow:
      1) OCR (optional) -> 2) Vision analysis -> 3) KB search -> 4) reflection x N -> return
    """
    # ensure pydantic or dict
    try:
        req = chat_request
        # if dict -> construct ChatRequest lazily
    except Exception:
        req = chat_request

    steps = []

    # 1) OCR: try extracting text if OCR model exists
    if req.image_path:
        try:
            ocr_in = OCRInput(image_path=req.image_path)
            ocr_out = run_ocr(ocr_in)
            steps.append(NodeOutput(node="ocr", content=ocr_out.text))
            # extend prompt with OCR text
        except Exception as e:
            steps.append(NodeOutput(node="ocr", content=f"ocr failed: {e}"))

    # 2) Vision analysis (DeepSeek-VL2)
    prompt = "この画像の人物の特徴・ポーズ・文化的モチーフを日本語で説明してください。"
    if req.text:
        # user text may be context or question; include it
        prompt = f"{req.text}\n\n{prompt}"
    img_in = ImageAnalysisInput(image_path=req.image_path, prompt=prompt)
    vis_out = run_image_analysis(img_in)
    steps.append(NodeOutput(node="vision", content=vis_out.summary))

    # 3) Knowledge search: use vision summary as query
    ks_in = KnowledgeSearchInput(query=vis_out.summary)
    ks_out = run_knowledge_search(ks_in)
    steps.append(NodeOutput(node="kb", content=str(len(ks_out.hits)) + " hits"))

    # 4) Reflection loop
    current = vis_out.summary
    for i in range(max(1, getattr(req, "reflection_rounds", 2))):
        refl_in = ReflectionInput(initial_explanation=current, hits=ks_out.hits)
        refl_out = run_reflection(refl_in)
        steps.append(NodeOutput(node=f"reflection_{i+1}", content=refl_out.refined))
        current = refl_out.refined

    # 5) Final answer: current
    resp = AgentResponse(final_answer=current, steps=steps)
    return resp

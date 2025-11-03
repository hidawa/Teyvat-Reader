"""
LangGraph版のエージェント実行グラフ。
画像処理・知識検索は除外し、テキスト入力のみをリフレクションノードで処理します。
"""

from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from typing import Any, List, TypedDict, Optional
from pydantic import BaseModel

from src.models.request_models import ChatRequest
from src.models.response_models import AgentResponse

from .nodes.reflection import ReflectionInput, run_reflection
# from .nodes.ocr_node import OCRInput, run_ocr
# from .nodes.image_analysis import ImageAnalysisInput, run_image_analysis


# --- 状態モデル -------------------------------------------------------------

class AgentState(BaseModel):
    """LangGraphの状態。"""
    input: str
    plans: List[str]
    feedbacks: List[str]
    output: Optional[str]
    iteration: int


# --- ノード定義 -------------------------------------------------------------

# OCR ノード（コメントアウト）
# @tool
# def ocr_node(state: AgentState) -> AgentState:
#     """OCR解析"""
#     ocr_in = OCRInput(image_path=state.image_path)
#     ocr_out = run_ocr(ocr_in)
#     state.steps.append(NodeOutput(node="ocr", content=ocr_out.text))
#     state.text += "\n" + ocr_out.text
#     return state


# Vision ノード（コメントアウト）
# @tool
# def vision_node(state: AgentState) -> AgentState:
#     """DeepSeek-VL2 画像解析"""
#     prompt = f"{state.text}\n\nこの画像の人物を日本語で説明してください。"
#     img_in = ImageAnalysisInput(image_path=state.image_path, prompt=prompt)
#     vis_out = run_image_analysis(img_in)
#     state.steps.append(NodeOutput(node="vision", content=vis_out.summary))
#     state.text += "\n" + vis_out.summary
#     return state


def reflection_node(state: AgentState) -> AgentState:
    """推論ノード（Reflection）"""
    refl_in = ReflectionInput(
        initial_explanation=state.input, 
        feedbacks=state.feedbacks
    )
    refl_out = run_reflection(refl_in)
    state.feedbacks.append(refl_out.refined)
    state.output = refl_out.refined
    # state.feedbacks.append(NodeOutput(node="reflection", content=refl_out.refined))
    return state


# --- グラフ構築 -------------------------------------------------------------

def build_workflow():
    
    workflow = StateGraph(AgentState)
    # 今はリフレクションノードだけ
    workflow.add_node("reflection", reflection_node)
    
    workflow.add_edge(START, "reflection")
    workflow.add_edge("reflection", END)
    # workflow.set_entry_point("reflection")

    return workflow


# --- 実行関数 ---------------------------------------------------------------

async def run_agent(chat_request: ChatRequest):
    """
    LangGraphを実行。画像や知識検索は行わず、テキストベースのリフレクションのみ。
    """
    state = AgentState(
        input=chat_request.text or "",
        plans=[],
        feedbacks=[],
        output=None,
        iteration=0,
    )

    workflow = build_workflow()
    runner = workflow.compile()

    final_state = await runner.ainvoke(state)
    resp = AgentResponse(final_answer=list(final_state.values())[0], steps=None)
    return resp

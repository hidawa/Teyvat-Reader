from fastapi import APIRouter, UploadFile, File
from src.agents.graph import run_teyvat_reader

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """画像を受け取り、LangGraph経由で解析・推論"""
    result = await run_teyvat_reader(file)
    return result
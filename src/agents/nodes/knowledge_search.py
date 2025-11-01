from pydantic import BaseModel
from typing import List, Dict, Any
import os, requests, json

# シンプルな入出力
class KnowledgeSearchInput(BaseModel):
    query: str

class KnowledgeSearchOutput(BaseModel):
    hits: List[Dict[str, Any]] = []

# 実装ポリシー：
# 1) 環境変数 SEARCH_API_URL / SEARCH_API_KEY が設定されていれば外部APIを呼ぶ
# 2) 無ければローカルの data/knowledge_base.json に対して単純キーワード検索を行う

KB_PATH = os.path.join(os.getcwd(), "data", "knowledge_base.json")

def run_knowledge_search(payload: KnowledgeSearchInput) -> KnowledgeSearchOutput:
    query = payload.query
    api_url = os.environ.get("SEARCH_API_URL")
    api_key = os.environ.get("SEARCH_API_KEY")
    if api_url and api_key:
        # 例：外部検索APIをPOSTで叩く想定（ユーザーで修正）
        res = requests.post(api_url, json={"q": query, "k": api_key}, timeout=10)
        try:
            data = res.json()
            return KnowledgeSearchOutput(hits=data.get("results", []))
        except Exception:
            return KnowledgeSearchOutput(hits=[])
    # ローカルKBフォールバック
    hits = []
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
            for item in kb:
                # 簡易マッチ（キーワードが含まれるか）
                if any(term.lower() in query.lower() for term in item.get("keywords", [])):
                    hits.append(item)
    return KnowledgeSearchOutput(hits=hits)

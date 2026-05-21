import requests

from pdf_rag.config import EMBEDDING_URL, EMBEDDING_MODEL, QIANFAN_API_KEY


def get_embedding(text):
    """调用千帆 Embedding API 获取文本向量"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QIANFAN_API_KEY}",
    }
    data = {
        "model": EMBEDDING_MODEL,
        "input": [text],
    }

    try:
        res = requests.post(EMBEDDING_URL, headers=headers, json=data, timeout=30).json()
        return res.get("data", [{}])[0].get("embedding")
    except Exception:
        return None

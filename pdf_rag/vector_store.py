import pickle
import os

import numpy as np
import faiss

from pdf_rag.config import (
    VECTOR_STORE_DIR,
    FAISS_INDEX_PATH,
    TEXTS_PATH,
    TOP_K,
)
from pdf_rag.embedding import get_embedding


def build_vector_store(text_chunks):
    """构建 FAISS 向量索引"""
    vectors = []
    valid_texts = []

    for chunk in text_chunks:
        emb = get_embedding(chunk)
        if emb is not None:
            vectors.append(emb)
            valid_texts.append(chunk)

    if not vectors:
        return None, None

    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, valid_texts


def save_vector_store(index, texts):
    """持久化向量库到本地"""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)


def load_vector_store():
    """从本地加载向量库"""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
    return index, texts


def similarity_search(question, index, texts, k=TOP_K):
    """向量检索 Top-K 相似文本块"""
    q_emb = get_embedding(question)
    if q_emb is None:
        return []

    q_emb = np.array([q_emb]).astype("float32")
    distances, indices = index.search(q_emb, k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": texts[idx],
            "score": float(distances[0][i]),
        })

    return results

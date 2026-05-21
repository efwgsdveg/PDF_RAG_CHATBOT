import logging
import pickle
import os
import time

import numpy as np
import faiss

from pdf_rag.config import (
    VECTOR_STORE_DIR,
    FAISS_INDEX_PATH,
    TEXTS_PATH,
    TOP_K,
)
from pdf_rag.embedding import get_embedding

logger = logging.getLogger(__name__)


def build_vector_store(text_chunks):
    """构建 FAISS 向量索引"""
    vectors = []
    valid_texts = []

    for i, chunk in enumerate(text_chunks):
        logger.debug("Embedding chunk %d/%d (%d chars)", i + 1, len(text_chunks), len(chunk))
        emb = get_embedding(chunk)
        if emb is not None:
            vectors.append(emb)
            valid_texts.append(chunk)
        else:
            logger.warning("chunk %d embedding 失败，已跳过", i)

    if not vectors:
        logger.error("所有 chunk embedding 均失败，向量库为空")
        return None, None

    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    logger.info("向量库构建完成: %d 个向量, 维度=%d", len(vectors), vectors.shape[1])
    return index, valid_texts


def save_vector_store(index, texts):
    """持久化向量库到本地"""
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)
    logger.info("向量库已保存: %d 个文本", len(texts))


def load_vector_store():
    """从本地加载向量库"""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(TEXTS_PATH, "rb") as f:
        texts = pickle.load(f)
    logger.info("向量库已加载: %d 个文本", len(texts))
    return index, texts


def similarity_search(question, index, texts, k=TOP_K):
    """向量检索 Top-K 相似文本块"""
    t0 = time.time()

    q_emb = get_embedding(question)
    if q_emb is None:
        logger.warning("检索失败：问题 embedding 为空")
        return []

    q_emb = np.array([q_emb]).astype("float32")
    distances, indices = index.search(q_emb, k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": texts[idx],
            "score": float(distances[0][i]),
        })

    elapsed = time.time() - t0
    scores = [r["score"] for r in results]
    logger.info("检索完成 | query=\"%s\" | top_k=%d | scores=%s | time=%.2fs",
                question[:50], k, [round(s, 3) for s in scores], elapsed)

    return results

import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from pdf_rag.config import EMBEDDING_URL, EMBEDDING_MODEL, QIANFAN_API_KEY, RETRY_MAX_ATTEMPTS, RETRY_MIN_DELAY, RETRY_MAX_DELAY

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MIN_DELAY, max=RETRY_MAX_DELAY),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False,
)
def _request_embedding(text):
    """带重试的 Embedding API 请求"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {QIANFAN_API_KEY}",
    }
    data = {
        "model": EMBEDDING_MODEL,
        "input": [text],
    }
    res = requests.post(EMBEDDING_URL, headers=headers, json=data, timeout=30)
    return res.json()


def get_embedding(text):
    """调用千帆 Embedding API 获取文本向量"""
    try:
        result = _request_embedding(text)
        emb = result.get("data", [{}])[0].get("embedding")
        if emb is None:
            logger.warning("Embedding API 返回异常结构: %s", result)
        return emb
    except Exception as e:
        logger.error("Embedding 请求全部重试后仍失败: %s", e)
        return None

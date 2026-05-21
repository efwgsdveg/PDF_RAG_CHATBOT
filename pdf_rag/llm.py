import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from pdf_rag.config import (
    LLM_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT,
    QIANFAN_API_KEY, RETRY_MAX_ATTEMPTS, RETRY_MIN_DELAY, RETRY_MAX_DELAY,
)

logger = logging.getLogger(__name__)


def _format_history(history):
    """将对话历史格式化为文本"""
    text = ""
    for msg in history[-4:]:
        role = "用户" if msg["role"] == "user" else "助手"
        text += f"{role}:{msg['content']}\n"
    return text


@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MIN_DELAY, max=RETRY_MAX_DELAY),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=False,
)
def _request_llm(prompt):
    """带重试的 LLM API 请求"""
    r = requests.post(
        LLM_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {QIANFAN_API_KEY}",
        },
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
        },
        timeout=LLM_TIMEOUT,
    )
    return r.json()


def ask_qianfan(question, context, history):
    """调用千帆 LLM 生成回答"""
    history_text = _format_history(history)

    prompt = f"""
你是一个严谨的PDF问答助手：

【规则】
1. 只能基于文档回答
2. 不允许编造
3. 找不到就说不知道

【历史对话】
{history_text}

【文档内容】
{context}

【问题】
{question}

【回答格式】
- 结论：
- 依据：
"""

    try:
        result = _request_llm(prompt)
        answer = result["choices"][0]["message"]["content"]
        logger.info("LLM 回答成功 | 输入长度=%d", len(prompt))
        return answer
    except Exception as e:
        logger.error("LLM 调用全部重试后仍失败: %s", e)
        return "⚠️ 模型调用失败，请稍后再试"


def rewrite_question(question, history):
    """根据对话历史补全问题（多轮对话支持）"""
    if not history:
        return question

    history_text = _format_history(history)

    prompt = f"""
根据对话，将问题补充完整：

{history_text}

问题：{question}
"""

    try:
        result = _request_llm(prompt)
        rewritten = result["choices"][0]["message"]["content"]
        logger.info("问题改写: %s -> %s", question, rewritten)
        return rewritten
    except Exception as e:
        logger.warning("问题改写失败，使用原始问题: %s", e)
        return question

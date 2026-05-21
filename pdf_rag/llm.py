import requests

from pdf_rag.config import LLM_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT


def _format_history(history):
    """将对话历史格式化为文本"""
    text = ""
    for msg in history[-4:]:
        role = "用户" if msg["role"] == "user" else "助手"
        text += f"{role}:{msg['content']}\n"
    return text


def ask_qianfan(api_key, question, context, history):
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
        r = requests.post(
            LLM_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": LLM_TEMPERATURE,
            },
            timeout=LLM_TIMEOUT,
        )
        result = r.json()
        return result["choices"][0]["message"]["content"]
    except Exception:
        return "⚠️ 模型调用失败，请稍后再试"


def rewrite_question(api_key, question, history):
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
        r = requests.post(
            LLM_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": LLM_TEMPERATURE,
            },
            timeout=LLM_TIMEOUT,
        )
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return question

# ------------------------------
# 导入库
# ------------------------------
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import pickle
import faiss
import os

# =========================================================
# 缓存Embedding
# =========================================================
@st.cache_data(show_spinner=False)
def get_embedding_cached(api_key, text):
    return get_embedding(api_key, text)

@st.cache_data(show_spinner=False)
def get_query_embedding_cached(api_key, question):
    return get_embedding(api_key, question)

# =========================================================
# 1 PDF读取
# =========================================================
def get_pdf_text(pdf_files):
    text = ""
    for file in pdf_files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# =========================================================
# 2 文本切分
# =========================================================
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# =========================================================
# 3 获取Embedding
# =========================================================
def get_embedding(api_key, text):
    url = "https://qianfan.baidubce.com/v2/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "embedding-v1",
        "input": [text]
    }

    try:
        res = requests.post(url, headers=headers, json=data, timeout=30).json()
        return res.get("data", [{}])[0].get("embedding")
    except Exception:
        return None

# =========================================================
# 4 构建向量库
# =========================================================
def build_vector_store(text_chunks, api_key):
    vectors, valid_texts = [], []

    for chunk in text_chunks:
        emb = get_embedding_cached(api_key, chunk)
        if emb is not None:
            vectors.append(emb)
            valid_texts.append(chunk)

    if not vectors:
        return None, None

    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, valid_texts

# =========================================================
# 5 保存向量库
# =========================================================
def save_vector_store(index, texts):
    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, "vector_store/faiss.index")

    with open("vector_store/texts.pkl", "wb") as f:
        pickle.dump(texts, f)

# =========================================================
# 6 加载向量库
# =========================================================
def load_vector_store():
    if not os.path.exists("vector_store/faiss.index"):
        return None

    index = faiss.read_index("vector_store/faiss.index")

    with open("vector_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    return index, texts

# =========================================================
# 7 相似度搜索
# =========================================================
def similarity_search(question, api_key, index, texts, k=5):

    q_emb = get_query_embedding_cached(api_key, question)
    if q_emb is None:
        return []

    q_emb = np.array([q_emb]).astype("float32")
    D, I = index.search(q_emb, k)

    results = []
    for i, idx in enumerate(I[0]):
        results.append({
            "text": texts[idx],
            "score": float(D[0][i])
        })

    return results

# =========================================================
# 8 LLM调用
# =========================================================
def ask_qianfan(api_key, question, context, history):

    url = "https://qianfan.baidubce.com/v2/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    history_text = ""
    for msg in history[-4:]:
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}:{msg['content']}\n"

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
        r = requests.post(url, headers=headers, json={
            "model": "ernie-3.5-8k",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }, timeout=30)

        result = r.json()
        return result["choices"][0]["message"]["content"]

    except Exception:
        return "⚠️ 模型调用失败，请稍后再试"

# =========================================================
# 问题改写
# =========================================================
def rewrite_question(api_key, question, history):

    if not history:
        return question

    history_text = ""
    for msg in history[-4:]:
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}:{msg['content']}\n"

    prompt = f"""
根据对话，将问题补充完整：

{history_text}

问题：{question}
"""

    try:
        r = requests.post(
            "https://qianfan.baidubce.com/v2/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "ernie-3.5-8k",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=30
        )
        return r.json()["choices"][0]["message"]["content"]

    except Exception:
        return question

# =========================================================
# 主程序
# =========================================================
def main():

    st.title("📄 PDF RAG Chatbot")

    if "vector_data" not in st.session_state:
        st.session_state.vector_data = load_vector_store()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        api_key = st.text_input("API Key", type="password")
        pdf_files = st.file_uploader("上传PDF", type="pdf", accept_multiple_files=True)

        if st.button("处理PDF"):
            if not api_key or not pdf_files:
                st.warning("请填写完整")
                return

            text = get_pdf_text(pdf_files)
            chunks = get_text_chunks(text)

            with st.spinner("处理中..."):
                index, texts = build_vector_store(chunks, api_key)

            if index is None:
                st.error("Embedding失败")
                return

            save_vector_store(index, texts)
            st.session_state.vector_data = (index, texts)
            st.success("完成")

        if st.button("🗑️ 清空对话"):
            st.session_state.chat_history = []
            st.rerun()

    # 显示历史
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📄 查看引用来源"):
                    for r in msg["sources"]:
                        st.write(r["text"][:200] + "...")

    # 输入
    question = st.chat_input("请输入问题")

    if question:

        if not api_key:
            st.warning("请输入 API Key")
            return

        if st.session_state.vector_data is None:
            st.warning("请先上传PDF")
            return

        index, texts = st.session_state.vector_data

        with st.chat_message("user"):
            st.write(question)

        st.session_state.chat_history.append({"role": "user", "content": question})

        new_q = rewrite_question(api_key, question, st.session_state.chat_history)

        results = similarity_search(new_q, api_key, index, texts)

        if not results:
            st.warning("未找到相关内容")
            return

        if any(word in question for word in ["总结", "概括"]):
            context = "\n".join(texts[:min(30, len(texts))])
        else:
            context = "\n".join([r["text"] for r in results[:5]])

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer = ask_qianfan(api_key, question, context, st.session_state.chat_history)

            st.write(answer)

            with st.expander("📄 查看引用来源"):
                for r in results:
                    st.write(r["text"][:200] + "...")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": results
        })


if __name__ == "__main__":
    main()
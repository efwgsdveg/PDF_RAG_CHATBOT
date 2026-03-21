# ------------------------------
# 导入库
# ------------------------------
import streamlit as st
import requests
from datetime import datetime

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import numpy as np
import pickle
import os


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
# 3 获取Embedding（千帆）
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
        res = requests.post(url, headers=headers, json=data).json()

        if "data" not in res:
            return None

        return res["data"][0]["embedding"]

    except Exception as e:
        return None


# =========================================================
# 4 构建向量库
# =========================================================
def build_vector_store(text_chunks, api_key):
    vectors = []
    valid_texts = []

    for chunk in text_chunks:
        emb = get_embedding(api_key, chunk)

        if emb is not None:
            vectors.append(emb)
            valid_texts.append(chunk)

    return np.array(vectors), valid_texts


# =========================================================
# 5 保存向量库
# =========================================================
def save_vector_store(vectors, texts):
    os.makedirs("vector_store", exist_ok=True)

    with open("vector_store/store.pkl", "wb") as f:
        pickle.dump((vectors, texts), f)


# =========================================================
# 6 加载向量库
# =========================================================
def load_vector_store():
    if not os.path.exists("vector_store/store.pkl"):
        return None

    with open("vector_store/store.pkl", "rb") as f:
        return pickle.load(f)


# =========================================================
# 7 余弦相似度
# =========================================================
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# =========================================================
# 8 相似度搜索
# =========================================================
def similarity_search(question, api_key, vectors, texts, top_k=10, final_k=3, min_score=0.2):

    q_emb = get_embedding(api_key, question)

    if q_emb is None:
        return []

    scores = []

    for v in vectors:
        score = cosine_similarity(q_emb, v)
        scores.append(score)

    scores = np.array(scores)

    # TopK
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for i in top_indices:
        if scores[i] > min_score:
            results.append({
                "text": texts[i],
                "score": float(scores[i])
            })

    results = results[:final_k]

    # ✅ 兜底机制（必须有）
    if not results and len(scores) > 0:
        best_idx = int(np.argmax(scores))
        results = [{
            "text": texts[best_idx],
            "score": float(scores[best_idx])
        }]

    return results


# =========================================================
# 9 千帆大模型
# =========================================================
def ask_qianfan(api_key, question, context, history):

    url = "https://qianfan.baidubce.com/v2/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    history_text = ""
    for q, a, _ in history[-3:]:
        history_text += f"用户:{q}\n助手:{a}\n"

    prompt = f"""
你是一个PDF文档助手。

历史对话:
{history_text}

请根据以下文档内容回答问题。
如果文档中没有答案，请回答：不知道。

文档内容:
{context}

问题:
{question}
"""

    data = {
        "model": "ernie-3.5-8k",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    r = requests.post(url, headers=headers, json=data)
    result = r.json()

    if "choices" not in result:
        return f"接口错误: {result}"

    return result["choices"][0]["message"]["content"]


# =========================================================
# 主程序
# =========================================================
def main():

    st.title("📄 PDF RAG Chatbot（Embedding版）")

    if "vector_data" not in st.session_state:
        st.session_state.vector_data = load_vector_store()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------- Sidebar ----------------
    with st.sidebar:

        api_key = st.text_input("千帆 API Key", type="password")

        pdf_files = st.file_uploader(
            "上传PDF",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("处理PDF"):

            if not api_key:
                st.warning("请输入 API Key")
                return

            if not pdf_files:
                st.warning("请上传PDF")
                return

            raw_text = get_pdf_text(pdf_files)

            chunks = get_text_chunks(raw_text)

            with st.spinner("生成Embedding（首次会较慢）..."):
                vectors, texts = build_vector_store(chunks, api_key)

            save_vector_store(vectors, texts)

            st.session_state.vector_data = (vectors, texts)

            st.success("处理完成")

            st.write("chunk数量:", len(chunks))
            st.write("向量数量:", len(vectors))

    # ---------------- Chat ----------------
    question = st.chat_input("请输入问题")

    if question:

        if not st.session_state.vector_data:
            st.warning("请先处理PDF")
            return

        vectors, texts = st.session_state.vector_data

        results = similarity_search(
            question,
            api_key,
            vectors,
            texts
        )

        docs = sorted(results, key=lambda x: x["score"], reverse=True)

        context = "\n".join([r["text"] for r in docs[:2]])

        answer = ask_qianfan(
            api_key,
            question,
            context,
            st.session_state.chat_history
        )

        with st.chat_message("assistant"):
            st.write(answer)

            st.markdown("**引用来源:**")

            for i, r in enumerate(results):
                st.markdown(f"来源 {i+1}（相似度: {r['score']:.4f}）")
                st.write(r["text"][:200] + "...")

        st.session_state.chat_history.append(
            (question, answer, datetime.now())
        )


if __name__ == "__main__":
    main()
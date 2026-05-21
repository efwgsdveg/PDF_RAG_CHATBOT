import logging
import os
import sys
import streamlit as st

from pdf_rag.pdf_utils import get_pdf_text, get_text_chunks
from pdf_rag.vector_store import (
    build_vector_store,
    save_vector_store,
    load_vector_store,
    similarity_search,
)
from pdf_rag.llm import ask_qianfan, rewrite_question
from pdf_rag.config import TOP_K, SUMMARY_TOP_K, QIANFAN_API_KEY


def main():
    # 日志配置
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler("logs/rag.log", encoding="utf-8"),
        ],
    )

    st.title("📄 PDF RAG Chatbot")

    # 启动时检查 API Key
    if not QIANFAN_API_KEY:
        st.error("未配置 QIANFAN_API_KEY，请在 .env 文件中设置或添加为环境变量")
        st.info("参考 .env.example 文件格式")
        return

    if "vector_data" not in st.session_state:
        st.session_state.vector_data = load_vector_store()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar
    with st.sidebar:
        pdf_files = st.file_uploader("上传PDF", type="pdf", accept_multiple_files=True)

        if st.button("处理PDF"):
            if not pdf_files:
                st.warning("请上传 PDF 文件")
                return

            text = get_pdf_text(pdf_files)
            chunks = get_text_chunks(text)
            logging.info("PDF 处理: %d 个文件 -> %d 个文本块", len(pdf_files), len(chunks))

            with st.spinner("处理中..."):
                index, texts = build_vector_store(chunks)

            if index is None:
                st.error("Embedding失败")
                return

            save_vector_store(index, texts)
            st.session_state.vector_data = (index, texts)
            st.success("完成")

        if st.button("🗑️ 清空对话"):
            st.session_state.chat_history = []
            st.rerun()

    # 历史对话
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📄 查看引用来源"):
                    for r in msg["sources"]:
                        st.write(r["text"][:200] + "...")

    # 用户输入
    question = st.chat_input("请输入问题")

    if question:
        if st.session_state.vector_data is None:
            st.warning("请先上传PDF")
            return

        index, texts = st.session_state.vector_data

        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # 问题改写（多轮对话）
        new_q = rewrite_question(question, st.session_state.chat_history)

        # 检索
        results = similarity_search(new_q, index, texts)
        if not results:
            st.warning("未找到相关内容")
            return

        # 区分总结类问题
        if any(word in question for word in ["总结", "概括"]):
            context = "\n".join(texts[: min(SUMMARY_TOP_K, len(texts))])
        else:
            context = "\n".join(r["text"] for r in results[:TOP_K])

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer = ask_qianfan(question, context, st.session_state.chat_history)

            st.write(answer)
            with st.expander("📄 查看引用来源"):
                for r in results:
                    st.write(r["text"][:200] + "...")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": results,
        })


if __name__ == "__main__":
    main()

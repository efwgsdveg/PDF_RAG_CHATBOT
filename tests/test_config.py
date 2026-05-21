from pdf_rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EMBEDDING_URL,
    EMBEDDING_MODEL,
    LLM_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    RETRY_MAX_ATTEMPTS,
)


class TestConfig:
    def test_embedding_config(self):
        assert EMBEDDING_URL == "https://qianfan.baidubce.com/v2/embeddings"
        assert EMBEDDING_MODEL == "embedding-v1"

    def test_llm_config(self):
        assert LLM_URL == "https://qianfan.baidubce.com/v2/chat/completions"
        assert LLM_MODEL == "ernie-3.5-8k"
        assert LLM_TEMPERATURE == 0.2
        assert LLM_TIMEOUT == 30

    def test_chunk_config(self):
        assert CHUNK_SIZE == 300
        assert CHUNK_OVERLAP == 50

    def test_retrieval_config(self):
        assert TOP_K == 5

    def test_retry_config(self):
        assert RETRY_MAX_ATTEMPTS == 3

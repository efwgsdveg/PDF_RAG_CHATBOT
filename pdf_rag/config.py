import os
from pathlib import Path


def _load_env():
    """手动加载 .env 文件，无需 python-dotenv 依赖"""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("\"'")
            if key and not os.getenv(key):
                os.environ[key] = value


_load_env()

# 千帆 API 密钥（优先级：系统环境变量 > .env 文件）
QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY", "")

# Embedding
EMBEDDING_URL = "https://qianfan.baidubce.com/v2/embeddings"
EMBEDDING_MODEL = "embedding-v1"

# LLM
LLM_URL = "https://qianfan.baidubce.com/v2/chat/completions"
LLM_MODEL = "ernie-3.5-8k"
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT = 30

# 文本切分
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# 检索
TOP_K = 5
SUMMARY_TOP_K = 30

# 向量库路径
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = "vector_store/faiss.index"
TEXTS_PATH = "vector_store/texts.pkl"

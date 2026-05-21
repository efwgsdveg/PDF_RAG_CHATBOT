# 配置常量

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

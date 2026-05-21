# 📄 PDF RAG Chatbot

基于 **RAG（Retrieval-Augmented Generation）** 的 PDF 智能问答系统，支持多轮对话、语义检索与上下文理解。

## 🚀 功能特点

- 📂 支持多 PDF 上传解析
- 🔍 基于 FAISS 的语义向量检索
- 🧠 多轮对话（上下文感知 + 问题重写）
- 📊 引用来源展示（附相似度评分）
- ⚡ 指数退避重试 + 结构化日志
- 🔧 pytest 单元测试覆盖

## 🧱 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit |
| 向量检索 | FAISS (L2 距离) |
| Embedding | 百度千帆 `embedding-v1` |
| LLM | 百度千帆 `ernie-3.5-8k` |
| 文本切分 | LangChain `RecursiveCharacterTextSplitter` |
| PDF 解析 | PyPDF2 |
| 测试 | pytest + mock |
| CI/CD | GitHub Actions |
| 部署 | Docker / docker-compose |

## 📁 项目结构

```
PDF_RAG_CHATBOT/
├── app.py                      # Streamlit UI 入口
├── pdf_rag/                    # 核心逻辑包
│   ├── config.py               # 集中配置（URL、模型名、chunk 大小等）
│   ├── pdf_utils.py            # PDF 读取 + 文本切分
│   ├── embedding.py            # Embedding API（含重试机制）
│   ├── vector_store.py         # FAISS 构建/持久化/检索
│   └── llm.py                  # LLM 问答 + 问题改写（含重试机制）
├── tests/                      # pytest 单元测试
│   ├── test_config.py
│   ├── test_pdf_utils.py
│   ├── test_embedding.py
│   ├── test_vector_store.py
│   └── test_llm.py
├── .env.example                # 配置模板
├── .github/workflows/ci.yml    # GitHub Actions 自动测试
├── Dockerfile                  # Docker 镜像构建
├── docker-compose.yml          # Docker 一键部署
└── requirements.txt
```

## 🧠 架构设计

```
用户问题
    ↓
[问题改写] ← 历史对话上下文
    ↓
[向量检索] ───→ FAISS 索引 (预构建)
    ↓
[Top-K 文本] ─→ 百度千帆 LLM
    ↓
[生成回答] ───→ 展示 + 引用来源
```

## 🖥️ 本地运行

### 1. 配置密钥

创建 `.env` 文件（参考 `.env.example`）：

```bash
QIANFAN_API_KEY=你的千帆API密钥
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动

```bash
streamlit run app.py
```

或使用虚拟环境：

```bash
source venv/Scripts/activate   # Windows: venv\Scripts\activate
streamlit run app.py
```

## 🐳 Docker 部署

```bash
# 构建并启动
docker compose up --build

# 后台运行
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

访问 `http://localhost:8501` 即可使用。

## 🧪 运行测试

```bash
pytest tests/ -v
```

## 🔄 CI/CD

每次推送到 `main` 分支，GitHub Actions 自动运行全部测试。

## ⚙️ 配置项

所有配置集中在 `pdf_rag/config.py`，包括：

- Embedding 模型与 API 地址
- LLM 模型、温度、超时
- 文本切分参数（chunk_size / overlap）
- 检索 Top-K 数量
- 重试策略（最大次数、间隔）

## 📷 示例

![alt text](image.png)

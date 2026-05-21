# syntax=docker/dockerfile:1
FROM python:3.10-slim

WORKDIR /app

# 先装依赖（利用 Docker 层缓存，不改 requirements.txt 就不会重装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY pdf_rag/ pdf_rag/
COPY app.py .
COPY .env.example .env.example

# Streamlit 配置
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# 容器启动时创建 .env 然后运行
CMD ["sh", "-c", "test -f .env || cp .env.example .env && streamlit run app.py"]

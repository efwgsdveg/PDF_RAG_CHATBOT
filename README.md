# PDF RAG 问答系统（Embedding版）

一个基于 PDF 文档的问答系统，采用 RAG（检索增强生成）方案，实现“上传文档 → 提问 → 智能回答”。

---

## 一、功能介绍

* 支持上传多个 PDF 文件
* 自动提取 PDF 文本内容
* 文本切分（避免上下文过长）
* 使用向量（Embedding）进行语义检索
* 调用大模型生成回答
* 支持简单多轮对话
* 显示引用来源（包含相似度）

---

## 二、技术方案

### 1. 整体流程

PDF → 文本提取 → 文本切分 → Embedding向量化 → 相似度检索 → 拼接上下文 → 大模型生成答案

---

### 2. 核心技术

* 前端框架：Streamlit
* PDF解析：PyPDF2
* 文本切分：LangChain（RecursiveCharacterTextSplitter）
* 向量计算：NumPy
* 向量生成：百度千帆 Embedding API
* 大模型：百度千帆 ERNIE-3.5
* 相似度计算：余弦相似度

---

## 三、项目结构

.
├── app.py                # 主程序
├── requirements.txt      # 依赖文件
├── vector_store/         # 本地向量缓存（不会上传）
└── README.md

---

## 四、安装依赖

pip install -r requirements.txt

---

## 五、运行项目

streamlit run app.py

---

## 六、使用说明

1. 在侧边栏输入千帆 API Key
2. 上传 PDF 文件
3. 点击「处理PDF」
4. 在输入框输入问题进行提问

---

## 七、注意事项

* 首次处理 PDF 时会调用 Embedding API，速度较慢
* 向量数据会缓存到本地 vector_store/
* 若删除该文件夹，需要重新处理 PDF
* 需要保证 API Key 有效，否则会返回接口错误

---


from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def get_pdf_text(pdf_files):
    """读取多个 PDF 并拼接为纯文本"""
    text = ""
    for file in pdf_files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    """将长文本切分成块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)

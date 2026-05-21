from pdf_rag.pdf_utils import get_text_chunks


class TestGetTextChunks:
    def test_empty_text(self):
        assert get_text_chunks("") == []

    def test_short_text_single_chunk(self):
        text = "Hello World"
        chunks = get_text_chunks(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        # 生成超过 chunk_size 的文本
        text = "hello world " * 100
        chunks = get_text_chunks(text)
        assert len(chunks) > 1
        # 每个 chunk 不超过 CHUNK_SIZE
        assert all(len(c) <= 310 for c in chunks)  # 300 + 50 overlap

    def test_chinese_text(self):
        text = "你好世界 " * 50
        chunks = get_text_chunks(text)
        assert len(chunks) >= 1
        assert all(len(c) > 0 for c in chunks)

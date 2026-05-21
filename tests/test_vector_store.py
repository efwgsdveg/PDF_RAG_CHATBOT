import os
import tempfile

import numpy as np
import faiss
import pytest

from pdf_rag.vector_store import save_vector_store, load_vector_store


class TestVectorStoreIO:
    @pytest.fixture(autouse=True)
    def setup(self):
        # 创建临时目录用于测试
        self.tmp_dir = tempfile.mkdtemp()
        self.orig_dir = "vector_store"
        yield
        # 清理
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def _build_dummy_index(self, dim=4, n=3):
        index = faiss.IndexFlatL2(dim)
        vectors = np.array([[1.0, 0, 0, 0],
                            [0, 1.0, 0, 0],
                            [0, 0, 1.0, 0]], dtype="float32")
        index.add(vectors)
        return index

    def test_save_and_load(self):
        index = self._build_dummy_index()
        texts = ["text1", "text2", "text3"]

        save_vector_store(index, texts)
        assert os.path.exists("vector_store/faiss.index")
        assert os.path.exists("vector_store/texts.pkl")

        loaded_index, loaded_texts = load_vector_store()
        assert loaded_texts == texts
        assert loaded_index.ntotal == 3

    def test_load_nonexistent(self):
        # 确保 vector_store 目录不存在时返回 None
        if os.path.exists("vector_store"):
            import shutil
            shutil.rmtree("vector_store")
        assert load_vector_store() is None

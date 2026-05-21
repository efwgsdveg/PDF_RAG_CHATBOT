from unittest.mock import patch, Mock

from pdf_rag.embedding import get_embedding


class TestGetEmbedding:
    @patch("pdf_rag.embedding._request_embedding")
    def test_success(self, mock_request):
        mock_request.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        result = get_embedding("test text")
        assert result == [0.1, 0.2, 0.3]

    @patch("pdf_rag.embedding._request_embedding")
    def test_api_returns_none(self, mock_request):
        mock_request.return_value = {"data": [{}]}
        result = get_embedding("test text")
        assert result is None

    @patch("pdf_rag.embedding._request_embedding")
    def test_api_returns_empty_data(self, mock_request):
        mock_request.return_value = {"data": []}
        result = get_embedding("test text")
        assert result is None

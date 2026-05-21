from unittest.mock import patch

from pdf_rag.llm import rewrite_question, ask_qianfan


class TestRewriteQuestion:
    def test_no_history(self):
        assert rewrite_question("净利润是多少", []) == "净利润是多少"

    @patch("pdf_rag.llm._request_llm")
    def test_fallback_on_api_failure(self, mock_request):
        mock_request.side_effect = Exception("API timeout")
        history = [
            {"role": "user", "content": "去年的年报讲了什么"},
            {"role": "assistant", "content": "讲了营收和利润"},
        ]
        # API 失败时回退到原始问题
        result = rewrite_question("净利润是多少", history)
        assert result == "净利润是多少"

    @patch("pdf_rag.llm._request_llm")
    def test_success_rewrite(self, mock_request):
        mock_request.return_value = {
            "choices": [{"message": {"content": "去年的年报净利润是多少"}}]
        }
        history = [
            {"role": "user", "content": "去年的年报讲了什么"},
            {"role": "assistant", "content": "讲了营收和利润"},
        ]
        result = rewrite_question("净利润是多少", history)
        assert result == "去年的年报净利润是多少"
        mock_request.assert_called_once()


class TestAskQianfan:
    @patch("pdf_rag.llm._request_llm")
    def test_success(self, mock_request):
        mock_request.return_value = {
            "choices": [{"message": {"content": "净利润是100亿"}}]
        }
        result = ask_qianfan("净利润是多少", "doc content", [])
        assert result == "净利润是100亿"

    @patch("pdf_rag.llm._request_llm")
    def test_api_failure_returns_fallback(self, mock_request):
        mock_request.side_effect = Exception("API timeout")
        result = ask_qianfan("净利润是多少", "doc content", [])
        assert "模型调用失败" in result

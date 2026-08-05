"""Ollama LLM 클라이언트 테스트.

Ollama는 로컬(127.0.0.1:11434) 또는 클라우드 원격 호스트의 OpenAI 호환 /generate,
/chat API를 사용한다. 이 테스트는 실제 Ollama 서버 없이도 동작하도록
requests를 mock으로 대체한다.
"""
import pytest

from vac.llm import OllamaClient, LLMConfig


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class FakeSession:
    """requests.Session 모의체: post 호출 기록 + 고정 응답."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        return self.responses.pop(0)


@pytest.fixture
def client(monkeypatch):
    """기본 127.0.0.1 로컬 클라이언트 + 세션 주입."""
    c = OllamaClient(LLMConfig(host="127.0.0.1", port=11434))
    monkeypatch.setattr(c, "_session", None)  # 생성자에서 만들지 않도록
    return c


def _inject(client, session):
    client._session = session
    return session


def test_default_host_and_port():
    cfg = LLMConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 11434


def test_generate_returns_text(monkeypatch):
    import requests
    c = OllamaClient(LLMConfig())
    sess = FakeSession([FakeResponse({"response": "안녕하세요", "done": True})])
    monkeypatch.setattr(requests, "Session", lambda: sess)
    out = c.generate("say hi")
    assert out == "안녕하세요"
    assert sess.calls[0]["json"]["prompt"] == "say hi"
    assert sess.calls[0]["json"]["stream"] is False


def test_generate_error_raises(monkeypatch):
    import requests
    c = OllamaClient(LLMConfig())
    sess = FakeSession([FakeResponse({"error": "boom"}, status=500)])
    monkeypatch.setattr(requests, "Session", lambda: sess)
    with pytest.raises(RuntimeError):
        c.generate("hi")


def test_chat_conversation(monkeypatch):
    """채팅 형식: messages 배열 전달."""
    import requests
    c = OllamaClient(LLMConfig())
    sess = FakeSession([FakeResponse({"message": {"content": "ok"}, "done": True})])
    monkeypatch.setattr(requests, "Session", lambda: sess)
    msgs = [{"role": "user", "content": "hello"}]
    out = c.chat(msgs)
    assert out == "ok"
    assert sess.calls[0]["json"]["messages"] == msgs


def test_cloud_override_base_url():
    """클라우드 모델: host override 시 base_url이 변경된다."""
    cfg = LLMConfig(host="https://mycloud.example.com", port=443)
    assert cfg.base_url == "https://mycloud.example.com"


def test_model_name_passthrough(monkeypatch):
    import requests
    c = OllamaClient(LLMConfig(model="llama3.1:8b"))
    sess = FakeSession([FakeResponse({"response": "x", "done": True})])
    monkeypatch.setattr(requests, "Session", lambda: sess)
    c.generate("hi")
    assert sess.calls[0]["json"]["model"] == "llama3.1:8b"

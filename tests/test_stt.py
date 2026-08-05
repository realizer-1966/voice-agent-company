"""STT(SpeechRecognizer) 래퍼 테스트.

Android SpeechRecognizer는 플랫폼 API라 proot 내부에 없으므로,
Hermes Android 브리지(127.0.0.1:1457)를 통해 RECOGNIZE_SPEECH 인텐트를
보내고 결과 텍스트를 돌려받는 방식으로 접근한다.

외부 의존성 없이 단위 테스트가 돌도록 HTTP 요청을 mock으로 대체한다.
"""
import pytest

from vac.stt import (
    AndroidSpeechRecognizer,
    STTConfig,
    STTResult,
    STTUnavailableError,
)


class FakeHTTP:
    """urllib.request.urlopen 모의체."""

    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self._status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        import json
        return json.dumps(self._payload).encode()

    @property
    def status(self):
        return self._status


class FakeURLOpen:
    def __init__(self, response):
        self._resp = response
        self.calls = []

    def __call__(self, url, data=None, timeout=None):
        self.calls.append((url, data))
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


@pytest.fixture
def recognizer():
    return AndroidSpeechRecognizer(STTConfig())


def test_default_config():
    cfg = STTConfig()
    assert cfg.bridge_url == "http://127.0.0.1:1457"
    assert cfg.language == "ko-KR"


def test_transcribe_success(monkeypatch, recognizer):
    """브리지가 recognized text를 반환하면 그대로 돌려준다."""
    import urllib.request
    fake = FakeURLOpen(FakeHTTP({"ok": True, "text": "안녕하세요"}))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = recognizer.transcribe()
    assert result.text == "안녕하세요"
    assert result.success is True
    # POST /intent 호출 확인
    url, _data = fake.calls[0]
    assert "/intent" in url.full_url
    assert b"RECOGNIZE_SPEECH" in url.data


def test_transcribe_no_result(monkeypatch, recognizer):
    """인식 결과가 없으면 STTResult(success=False) 반환."""
    import urllib.request
    fake = FakeURLOpen(FakeHTTP({"ok": True}))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    result = recognizer.transcribe()
    assert result.success is False
    assert result.text == ""


def test_transcribe_error_raises(monkeypatch, recognizer):
    """브리지가 오류를 반환하면 STTUnavailableError 발생."""
    import urllib.request
    fake = FakeURLOpen(FakeHTTP({"ok": False, "error": "no speech service"}, status=500))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(STTUnavailableError):
        recognizer.transcribe()


def test_transcribe_network_error(monkeypatch, recognizer):
    """브리지 연결 실패 시 STTUnavailableError 발생."""
    import urllib.request
    fake = FakeURLOpen(OSError("connection refused"))
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    with pytest.raises(STTUnavailableError):
        recognizer.transcribe()

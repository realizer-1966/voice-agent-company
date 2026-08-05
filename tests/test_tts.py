"""TTS(SuperTonic) 래퍼 테스트.

SuperTonic은 온디바이스 ONNX TTS로, `/tmp/supertonic/py/helper.py`의
`load_text_to_speech` 를 통해 4-모델 ONNX 파이프라인을 로드하고
44.1kHz WAV를 생성한다. 실제 모델/추론 없이도 테스트가 돌도록
SuperTonic 파이프라인 로더는 import 시점이 아닌 실행 시점에 느리게
import하며, 테스트에서는 이 로더를 mock으로 대체한다.
"""
import numpy as np
import pytest

from vac.tts import (
    SuperTonicTTS,
    TTSError,
    TTSConfig,
)


class FakeTTS:
    """helper.load_text_to_speech()가 돌려주는 모의 파이프라인."""

    sample_rate = 44100

    def __init__(self, wav=None):
        self.wav = wav if wav is not None else np.zeros((1, 44100), dtype=np.float32)
        self.calls = []

    def __call__(self, text, lang, style, total_step=8, speed=1.0):
        self.calls.append((text, lang, style, total_step, speed))
        return self.wav, len(self.wav[0]) / self.sample_rate


def test_default_config():
    cfg = TTSConfig()
    assert cfg.onnx_dir == ""
    assert cfg.style_path == ""
    assert cfg.language == "ko"


def test_tts_loads_pipeline_lazily(monkeypatch):
    """synthesize 호출 시점에만 supertonic helper 를 import/로드한다."""
    fake = FakeTTS()
    calls = {}

    class FakeModule:
        def load_text_to_speech(self, *a, **k):
            calls["loaded"] = (a, k)
            return fake
        def load_voice_style(self, *a, **k):
            calls["style"] = a
            return {"style": "x"}

    import sys
    monkeypatch.setitem(sys.modules, "helper", FakeModule())

    tts = SuperTonicTTS(TTSConfig(onnx_dir="/models", style_path="/m.json"))
    # 아직 로드 안 됨
    assert tts._pipeline is None
    result = tts.synthesize("안녕하세요")
    assert result["audio"].shape == (1, 44100)
    assert result["sample_rate"] == 44100
    assert result["duration_sec"] == pytest.approx(1.0)
    # lazy 로드 확인
    assert calls["loaded"]
    assert calls["loaded"][0][0] == "/models"


def test_synthesize_returns_wav_bytes(monkeypatch):
    """16-bit WAV 바이트 스트림 생성."""
    fake = FakeTTS()
    class FakeModule:
        def load_text_to_speech(self, *a, **k):
            return fake
        def load_voice_style(self, *a, **k):
            return {"style": "x"}
    import sys
    monkeypatch.setitem(sys.modules, "helper", FakeModule())

    tts = SuperTonicTTS(TTSConfig(onnx_dir="/models"))
    wav_bytes = tts.synthesize("hi", as_wav=True)
    # RIFF/WAVE 헤더
    assert wav_bytes[:4] == b"RIFF"
    assert wav_bytes[8:12] == b"WAVE"


def test_tts_error_when_model_missing(monkeypatch):
    """모델 로드 실패 시 TTSError 발생."""
    class FakeModule:
        def load_text_to_speech(self, *a, **k):
            raise FileNotFoundError("model.onnx not found")
        def load_voice_style(self, *a, **k):
            raise FileNotFoundError("style.json not found")
    import sys
    monkeypatch.setitem(sys.modules, "helper", FakeModule())
    tts = SuperTonicTTS(TTSConfig(onnx_dir="/nonexistent"))
    with pytest.raises(TTSError):
        tts.synthesize("hi")

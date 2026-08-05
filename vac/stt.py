"""vac.stt — Android SpeechRecognizer 래퍼.

SpeechRecognizer는 Android 플랫폼의 온디바이스 음성인식 API다. proot
환경(Android 위)에서는 Hermes Android 브리지(127.0.0.1:1457)를 통해
`android.speech.action.RECOGNIZE_SPEECH` 인텐트를 호출하고,
`RecognizerIntent.EXTRA_RESULTS` 로 결과 텍스트를 받아온다.

단위 테스트는 urllib.request.urlopen 을 mock으로 대체해 외부 서버 없이
돌아가도록 한다.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Optional


class STTUnavailableError(RuntimeError):
    """음성인식 서비스 접근 불가/오류."""


@dataclass
class STTConfig:
    bridge_url: str = "http://127.0.0.1:1457"
    language: str = "ko-KR"
    timeout: float = 30.0


@dataclass
class STTResult:
    text: str
    success: bool
    confidence: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.success and bool(self.text)


class AndroidSpeechRecognizer:
    """Android SpeechRecognizer → Hermes 브리지 래퍼."""

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()

    def transcribe(self) -> STTResult:
        """마이크 입력을 받아 인식된 텍스트를 돌려준다.

        브리지의 /intent 로 RECOGNIZE_SPEECH 인텐트를 보내고, 응답 JSON의
        'text' 필드를 읽는다. 실패/연결 불가 시 STTUnavailableError 를 던진다.
        """
        payload = {
            "action": "android.speech.action.RECOGNIZE_SPEECH",
            "extras": {
                "android.speech.extra.LANGUAGE": self.config.language,
                "android.speech.extra.LANGUAGE_MODEL": "free_form",
            },
        }
        url = f"{self.config.bridge_url}/intent"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:  # 연결 실패, timeout, parse 오류 등
            raise STTUnavailableError(
                f"SpeechRecognizer bridge unavailable: {exc}"
            ) from exc

        if not data.get("ok"):
            raise STTUnavailableError(data.get("error", "unknown STT error"))

        text = data.get("text") or ""
        if not text:
            return STTResult(text="", success=False)
        return STTResult(text=text, success=True)

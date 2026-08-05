"""vac.llm — Ollama LLM 클라이언트.

로컬(127.0.0.1:11434)과 클라우드(원격 host) Ollama 서버를 모두 지원한다.
Ollama는 OpenAI 호환 `/generate`, `/chat` 엔드포인트를 제공하므로
requests 기반으로 단순하게 구현한다. 실제 서버 없이도 단위 테스트가
돌도록 세션은 요청 시 생성한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests


@dataclass
class LLMConfig:
    """Ollama 연결 설정.

    host가 "127.0.0.1"/"localhost"이면 로컬, 그 외(도메인/IP)는 클라우드로 취급.
    """
    host: str = "127.0.0.1"
    port: int = 11434
    model: str = "qwen3:0.6b"
    timeout: float = 120.0

    @property
    def base_url(self) -> str:
        scheme = "https" if self.port == 443 else "http"
        host = self.host.rstrip("/")
        if host.startswith("http"):
            return host
        return f"{scheme}://{host}:{self.port}"

    @property
    def is_local(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")


class OllamaClient:
    """Ollama /generate, /chat 클라이언트."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._session: Optional[requests.Session] = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def generate(self, prompt: str, *, model: Optional[str] = None,
                 system: Optional[str] = None) -> str:
        """비스트리밍 단일 프롬프트 생성. 생성된 텍스트를 반환."""
        payload: dict = {
            "model": model or self.config.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        resp = self.session.post(
            f"{self.config.base_url}/api/generate", json=payload,
            timeout=self.config.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama generate failed: HTTP {resp.status_code}: "
                               f"{resp.json().get('error', '')}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Ollama generate error: {data['error']}")
        return data.get("response", "")

    def chat(self, messages: list[dict], *, model: Optional[str] = None,
             system: Optional[str] = None) -> str:
        """대화 형식 생성. messages는 [{role, content}, ...]."""
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        payload: dict = {
            "model": model or self.config.model,
            "messages": msgs,
            "stream": False,
        }
        resp = self.session.post(
            f"{self.config.base_url}/api/chat", json=payload,
            timeout=self.config.timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama chat failed: HTTP {resp.status_code}: "
                               f"{resp.json().get('error', '')}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"Ollama chat error: {data['error']}")
        return data.get("message", {}).get("content", "")

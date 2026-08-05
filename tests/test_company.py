"""에이전트 회사 코어 테스트.

역할(PM/Dev/QA/Reviewer) 정의와 음성 대화 루프를 검증한다.
STT/TTS/LLM 은 모두 추상 의존성(프로토콜)으로 주입하므로
mock으로 대체해 회사 로직만 단위 테스트한다.
"""
import pytest

from vac.company import (
    AgentRole,
    Company,
    CompanyConfig,
    ROLE_SYSTEM_PROMPTS,
)
from vac.stt import STTResult


class FakeLLM:
    """OllamaClient 와 동일한 chat() 시그니처를 가진 모의체."""

    def __init__(self, responses=None):
        self.responses = list(responses or ["회사 응답"])
        self.calls = []

    def chat(self, messages, *, model=None, system=None):
        self.calls.append({"messages": messages, "system": system})
        return self.responses.pop(0)


class FakeSTT:
    def __init__(self, texts=None):
        self.texts = list(texts) if texts is not None else ["안녕하세요"]
        self.transcribe_calls = 0

    def transcribe(self):
        self.transcribe_calls += 1
        if not self.texts:
            return STTResult("", success=False)
        return STTResult(self.texts.pop(0), success=True)


class FakeTTS:
    def __init__(self):
        self.synthesized = []

    def synthesize(self, text, *, as_wav=False):
        self.synthesized.append(text)
        return {"audio": [0.0], "sample_rate": 44100, "duration_sec": 1.0}


def test_role_system_prompts_exist():
    for role in AgentRole:
        assert ROLE_SYSTEM_PROMPTS[role]
        assert role.name in ROLE_SYSTEM_PROMPTS[role]


def test_company_roles_registered():
    cfg = CompanyConfig()
    assert AgentRole.PM in cfg.roles
    assert AgentRole.DEV in cfg.roles
    assert AgentRole.QA in cfg.roles
    assert AgentRole.REVIEWER in cfg.roles


def test_dispatch_forwards_to_pm(monkeypatch):
    """사용자 지시가 특정 역할에 라우팅되면 해당 역할로 전달된다."""
    llm = FakeLLM(["PM가 처리하겠습니다."])
    company = Company(llm=llm, stt=FakeSTT(), tts=FakeTTS())
    reply = company.dispatch("DEV", "새 웹앱을 만들어줘")
    assert reply == "PM가 처리하겠습니다."
    assert llm.calls[0]["system"] == ROLE_SYSTEM_PROMPTS[AgentRole.DEV]


def test_voice_conversation_roundtrip():
    """음성 → 텍스트(STT) → LLM → 텍스트 응답 → TTS(음성) 왕복."""
    llm = FakeLLM(["개발 회사가 답합니다."])
    stt = FakeSTT(["웹앱 만들어줘"])
    tts = FakeTTS()
    company = Company(llm=llm, stt=stt, tts=tts)

    reply = company.voice_chat()
    assert reply == "개발 회사가 답합니다."
    assert stt.transcribe_calls == 1
    assert llm.calls[0]["messages"][0]["content"] == "웹앱 만들어줘"
    # TTS 로 응답이 음성 합성됨
    assert tts.synthesized[-1] == "개발 회사가 답합니다."


def test_voice_chat_no_speech(monkeypatch):
    """인식된 음성이 없으면 None 반환(무음 대응)."""
    llm = FakeLLM()
    stt = FakeSTT([])
    tts = FakeTTS()
    company = Company(llm=llm, stt=stt, tts=tts)
    assert company.voice_chat() is None
    assert llm.calls == []
    assert tts.synthesized == []

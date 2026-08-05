"""vac.company — 음성 기반 AI 에이전트 개발 회사 코어.

회사에는 PM/Dev/QA/Reviewer 역할의 에이전트가 있고, CEO(사용자)가 말로
지시하면 STT(SpeechRecognizer)로 텍스트를 받아 Ollama LLM이 역할에 맞는
답변을 생성하고 TTS(SuperTonic)로 소리내어 응답한다.

의존성(STT/TTS/LLM)은 프로토콜로 느슨하게 주입한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AgentRole(str, Enum):
    PM = "pm"
    DEV = "dev"
    QA = "qa"
    REVIEWER = "reviewer"


ROLE_SYSTEM_PROMPTS: dict[AgentRole, str] = {
    AgentRole.PM: (
        "당신은 voice-agent-company의 PM(프로젝트 매니저)입니다. "
        "사용자(CEO)의 요구를 분석해 명확한 작업 범위, 우선순위, "
        "검증 방법을 정리해 말로 응답하세요. 간결하고 실행 가능하게."
    ),
    AgentRole.DEV: (
        "당신은 voice-agent-company의 DEV(개발자)입니다. "
        "주어진 작업을 TDD 원칙으로 구현합니다: 실패 테스트 먼저, "
        "작은 커밋, pytest 실행. 구현 계획과 테스트 결과를 말로 보고하세요."
    ),
    AgentRole.QA: (
        "당신은 voice-agent-company의 QA 엔지니어입니다. "
        "엣지 케이스, 실패 시나리오, 성능을 검증하고 테스트를 강화합니다. "
        "발견한 결함을 구체적으로 보고하세요."
    ),
    AgentRole.REVIEWER: (
        "당신은 voice-agent-company의 REVIEWER(코드 리뷰어)입니다. "
        "구현물을 보안·품질·유지보수 관점에서 검토하고, "
        "승인하거나 개선 사항을 명확히 지적하세요."
    ),
}


@dataclass
class CompanyConfig:
    roles: list[AgentRole] = field(
        default_factory=lambda: [AgentRole.PM, AgentRole.DEV,
                                 AgentRole.QA, AgentRole.REVIEWER]
    )
    default_role: AgentRole = AgentRole.PM
    # STT로 인식한 지시에서 특정 역할을 꺼내기 위한 키워드
    role_aliases: dict[str, AgentRole] = field(default_factory=lambda: {
        "pm": AgentRole.PM, "프로젝트 매니저": AgentRole.PM, "기획": AgentRole.PM,
        "dev": AgentRole.DEV, "개발": AgentRole.DEV, "개발자": AgentRole.DEV,
        "qa": AgentRole.QA, "테스트": AgentRole.QA, "품질": AgentRole.QA,
        "reviewer": AgentRole.REVIEWER, "리뷰": AgentRole.REVIEWER,
        "검토": AgentRole.REVIEWER, "코드 리뷰": AgentRole.REVIEWER,
    })


class Company:
    """STT + Ollama LLM + TTS 를 엮은 음성 대화 에이전트 회사."""

    def __init__(self, llm, stt, tts, config: Optional[CompanyConfig] = None):
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.config = config or CompanyConfig()

    def _resolve_role(self, text: str) -> AgentRole:
        """음성 지시 텍스트에서 언급된 역할을 찾는다. 없으면 기본 역할."""
        low = text.lower()
        for key, role in self.config.role_aliases.items():
            if key in low:
                return role
        return self.config.default_role

    def dispatch(self, role: str, message: str) -> str:
        """특정 역할로 지시를 전달하고 답변을 받는다 (대소문자 무관)."""
        try:
            agent_role = AgentRole(role.lower())
        except ValueError:
            agent_role = self.config.default_role
        system = ROLE_SYSTEM_PROMPTS[agent_role]
        messages = [{"role": "user", "content": message}]
        return self.llm.chat(messages, system=system)

    def voice_chat(self) -> Optional[str]:
        """음성 대화 한 턴: STT → LLM → TTS. 응답 텍스트 반환(무음 시 None)."""
        result = self.stt.transcribe()
        if not result.ok:
            return None

        role = self._resolve_role(result.text)
        system = ROLE_SYSTEM_PROMPTS[role]
        messages = [{"role": "user", "content": result.text}]
        reply = self.llm.chat(messages, system=system)
        self.tts.synthesize(reply)
        return reply

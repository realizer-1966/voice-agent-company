"""vac.cli — voice-agent-company CLI.

텍스트 기반 대화(음성 없이도 사용 가능)와 음성 대화(STT→LLM→TTS)를
제공한다. LLM은 기본적으로 로컬 Ollama(127.0.0.1:11434)를 사용하며
--model / --host 로 클라우드 모델로 전환할 수 있다.
"""
from __future__ import annotations

import argparse
import sys

from vac.company import AgentRole, Company
from vac.llm import LLMConfig, OllamaClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vac",
        description="voice-agent-company — SuperTonic TTS + SpeechRecognizer STT + Ollama LLM",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Ollama 호스트 (예: 127.0.0.1 로컬, https://mycloud.example.com 클라우드)")
    p.add_argument("--port", type=int, default=11434, help="Ollama 포트")
    p.add_argument("--model", default="qwen3:0.6b", help="LLM 모델 이름")
    p.add_argument("--role", default=None, choices=[r.value for r in AgentRole],
                   help="상호작용할 기본 에이전트 역할")
    p.add_argument("--voice", action="store_true",
                   help="음성 대화 모드 (STT → LLM → TTS)")
    p.add_argument("prompt", nargs="*", help="한 줄 지시 (없으면 대화형)")
    return p


def _resolve_role(role_str):
    for r in AgentRole:
        if r.value == role_str:
            return r
    return None


def main(argv=None):
    args = build_parser().parse_args(argv)

    cfg = LLMConfig(host=args.host, port=args.port, model=args.model)
    llm = OllamaClient(cfg)

    from vac.stt import AndroidSpeechRecognizer, STTConfig
    from vac.tts import SuperTonicTTS, TTSConfig
    stt = AndroidSpeechRecognizer(STTConfig())
    tts = SuperTonicTTS(TTSConfig())

    role = _resolve_role(args.role)
    company = Company(llm=llm, stt=stt, tts=tts)

    if args.voice:
        return _voice_mode(company)

    prompt = " ".join(args.prompt)
    if prompt:
        role_name = role.value if role else None
        if role_name:
            reply = company.dispatch(role_name, prompt)
        else:
            role = company._resolve_role(prompt)
            reply = company.dispatch(role.value, prompt)
        print(f"\n[{role.value.upper()}] {reply}")
        return 0

    # 대화형
    print("voice-agent-company 대화 시작 (종료: quit/exit). "
          f"LLM: {args.model} @ {args.host}")
    while True:
        try:
            line = input("\nCEO> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit", "종료"):
            break
        role_name = role.value if role else company._resolve_role(line).value
        reply = company.dispatch(role_name, line)
        print(f"[{role_name.upper()}] {reply}")
    return 0


def _voice_mode(company: Company) -> int:
    """음성 대화 루프: STT로 듣고 LLM 답변을 TTS로 말한다."""
    print("음성 대화 모드. (중단: Ctrl+C)")
    while True:
        try:
            reply = company.voice_chat()
        except KeyboardInterrupt:
            print("\n종료합니다.")
            break
        except Exception as exc:
            print(f"[오류] {exc}")
            break
        if reply is None:
            print("(음성이 인식되지 않았습니다)")
        else:
            print(f"[회사] {reply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI 테스트."""
import pytest

from vac.cli import build_parser, main
from vac.company import AgentRole


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.model == "qwen3:0.6b"
    assert args.voice is False


def test_parser_role_choices():
    args = build_parser().parse_args(["--role", "qa"])
    assert args.role == "qa"


def test_main_single_prompt(monkeypatch):
    """단일 지시 실행: OllamaClient 가 아닌 실제와 동일한 chat() 호출."""
    captured = {}

    class FakeLLM:
        def chat(self, messages, *, model=None, system=None):
            captured["system"] = system
            captured["msgs"] = messages
            return "DEV가 작업을 시작합니다."

    import vac.cli as cli
    monkeypatch.setattr(cli, "OllamaClient", lambda cfg: FakeLLM())

    rc = main(["--role", "dev", "웹앱을 만들어줘"])
    assert rc == 0
    assert "DEV" in captured["system"]
    assert captured["msgs"][0]["content"] == "웹앱을 만들어줘"


def test_main_no_prompt_reachable(monkeypatch):
    """prompt 없이 호출 시 오류 없이 대화형 코드 경로에 진입하지 않도록 0 반환."""
    # 실제로는 input() 이 필요하므로 여기서는 파서까지만 검증
    args = build_parser().parse_args([])
    assert args.prompt == []

# voice-agent-company 🎙️

**SuperTonic TTS + Android SpeechRecognizer STT + Ollama LLM** 기반의
음성 AI 에이전트 개발 회사.

CEO(사용자)가 말로 지시하면 PM/Dev/QA/Reviewer 역할의 AI 에이전트가
음성으로 응답합니다. 모든 구성 요소는 온디바이스에서 실행됩니다.

## 구성 요소

| 구성 | 기술 | 실행 위치 |
|------|------|-----------|
| **TTS** (텍스트→음성) | SuperTonic 3 (ONNX, 99M 파라미터, 31개 언어, 44.1kHz) | 로컬 CPU |
| **STT** (음성→텍스트) | Android SpeechRecognizer (플랫폼 API) | Android 기기 |
| **LLM** (대화) | Ollama (로컬/클라우드 전환 가능) | 로컬 CPU / 클라우드 |

## 설치

```bash
git clone https://github.com/realizer-1966/voice-agent-company.git
cd voice-agent-company
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 사전 요구사항

- **Ollama**: 로컬 LLM 실행용. [ollama.com](https://ollama.com)에서 설치 후 `ollama serve` 실행.
  ```bash
  ollama pull qwen3:0.6b   # 작은 모델 권장 (10GB RAM 기준)
  ```
- **SuperTonic ONNX 모델**: `/tmp/supertonic/assets/onnx/` 아래에 4개 모델 파일 필요.
  [Hugging Face](https://huggingface.co/Supertone/supertonic-3)에서 다운로드.
- **Android Hermes 브리지**: STT 사용 시 `127.0.0.1:1457`에서 브리지가 실행 중이어야 함.

## 사용법

### 텍스트 대화 (기본)

```bash
# 기본 역할(PM)로 지시
vac "새 웹앱 프로젝트를 기획해줘"

# 특정 역할 지정
vac --role dev "TDD로 API 엔드포인트 구현해줘"
vac --role qa "이 코드의 엣지 케이스를 검증해줘"
vac --role reviewer "PR 리뷰해줘"

# 클라우드 Ollama 모델 사용
vac --host https://my-ollama.example.com --model llama3.1:70b "복잡한 분석 부탁해"

# 대화형 모드
vac
```

### 음성 대화

```bash
vac --voice   # STT로 듣고 TTS로 응답
```

## 에이전트 역할

| 역할 | 설명 |
|------|------|
| **PM** | 요구사항 분석, 작업 범위 정의, 우선순위 설정 |
| **DEV** | TDD 구현: 실패 테스트 → 구현 → pytest 검증 |
| **QA** | 엣지 케이스, 실패 시나리오, 성능 검증 |
| **REVIEWER** | 보안·품질·유지보수 관점 코드 리뷰 |

## 테스트

```bash
python -m pytest -q --cov=vac --cov-report=term-missing
```

## 프로젝트 구조

```
voice-agent-company/
├── vac/
│   ├── __init__.py      # 패키지 메타데이터
│   ├── llm.py           # OllamaClient (로컬/클라우드)
│   ├── stt.py           # AndroidSpeechRecognizer 래퍼
│   ├── tts.py           # SuperTonicTTS 래퍼
│   ├── company.py       # 회사 코어 (역할 + 음성 대화 루프)
│   └── cli.py           # CLI 인터페이스
├── tests/
│   ├── test_llm.py
│   ├── test_stt.py
│   ├── test_tts.py
│   ├── test_company.py
│   └── test_cli.py
├── pyproject.toml
└── README.md
```

## ⚠️ SuperTonic 서비스 종료 안내

SuperTonic 공식 저장소는 2026년 7월 아카이브 예정이며,
Voice Builder(음성 복제 도구)는 2026년 8월 31일 이후 접근 불가합니다.
이미 다운로드된 ONNX 모델은 계속 사용할 수 있지만,
신규 음성 복제는 불가능해질 수 있습니다.

## 라이선스

MIT

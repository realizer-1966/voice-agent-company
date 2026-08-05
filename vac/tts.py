"""vac.tts — SuperTonic 온디바이스 TTS 래퍼.

SuperTonic(Supertone)은 ONNX Runtime 기반의 멀티링궐(31개) 온디바이스
TTS다. 4-모델 파이프라인(딥포즈, 텍스트 인코더, 벡터 추정, vocoder)을
로드해 44.1kHz 16-bit WAV를 생성한다.

이 모듈은 `helper.py`(SuperTonic 저장소의 py/helper.py)를 *실행 시점에*
느리게 import한다. 그래서 모델/onnxruntime이 없는 환경에서도 vac 패키지
자체는 import 가능하며, synthesize() 호출 시에만 SuperTonic 의존성이
필요하다. 단위 테스트는 이 helper 를 mock으로 대체한다.
"""
from __future__ import annotations

import io
import struct
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


class TTSError(RuntimeError):
    """TTS 로드/합성 실패."""


@dataclass
class TTSConfig:
    onnx_dir: str = ""          # SuperTonic ONNX 모델 디렉터리
    style_path: str = ""        # voice style JSON 경로 (예: .../M1.json)
    language: str = "ko"        # ko / en / na(언어 무관)
    total_step: int = 8
    speed: float = 1.0
    use_gpu: bool = False


class SuperTonicTTS:
    """SuperTonic TTS 파이프라인 래퍼 (lazy load)."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._pipeline = None
        self._style = None

    def _load(self):
        """helper.py 를 import 하고 SuperTonic 파이프라인을 로드한다."""
        try:
            sys.path.insert(0, "/tmp/supertonic/py")
            import helper  # SuperTonic 저장소의 py/helper.py
        except ImportError as exc:
            raise TTSError(
                "SuperTonic helper.py를 찾을 수 없습니다. "
                "/tmp/supertonic 저장소를 먼저 준비하세요."
            ) from exc

        if not self.config.onnx_dir:
            raise TTSError("onnx_dir(TTSConfig.onnx_dir)이 비어 있습니다.")

        try:
            self._pipeline = helper.load_text_to_speech(
                self.config.onnx_dir, use_gpu=self.config.use_gpu
            )
        except Exception as exc:
            raise TTSError(f"SuperTonic 파이프라인 로드 실패: {exc}") from exc

        style_paths = []
        if self.config.style_path:
            style_paths = [self.config.style_path]
        try:
            self._style = helper.load_voice_style(style_paths)
        except Exception as exc:
            raise TTSError(f"voice style 로드 실패: {exc}") from exc

    @property
    def sample_rate(self) -> int:
        return self._pipeline.sample_rate if self._pipeline else 44100

    def synthesize(self, text: str, *, as_wav: bool = False) -> dict:
        """텍스트를 음성으로 합성.

        as_wav=False → {"audio": float32 ndarray (1, N), "sample_rate": int,
                        "duration_sec": float}
        as_wav=True  → {"wav": 16-bit PCM WAV bytes, ...} 추가
        """
        if self._pipeline is None:
            self._load()

        lang = self.config.language
        if text.isascii():
            lang = "en"

        try:
            wav, duration = self._pipeline(
                text, lang, self._style,
                total_step=self.config.total_step,
                speed=self.config.speed,
            )
        except Exception as exc:
            raise TTSError(f"SuperTonic 합성 실패: {exc}") from exc

        wav = np.asarray(wav, dtype=np.float32)
        # SuperTonic의 dur는 numpy 배열일 수 있으므로 스칼라로 안전 변환
        try:
            dur_sec = float(np.asarray(duration).reshape(-1)[0])
        except (TypeError, IndexError, ValueError):
            dur_sec = wav.shape[1] / self.sample_rate
        result: dict = {
            "audio": wav,
            "sample_rate": self.sample_rate,
            "duration_sec": dur_sec,
        }
        if as_wav:
            return _to_wav_bytes(wav, self.sample_rate)
        return result


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """float32 (1,N) 또는 (N,) 배열을 16-bit PCM WAV 바이트로 변환."""
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio[0]
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    num_channels = 1
    bits_per_sample = 16
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    data_size = len(pcm) * bytes_per_sample

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate,
                          byte_rate, block_align, bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()

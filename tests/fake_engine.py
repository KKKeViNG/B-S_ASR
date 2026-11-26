"""Deterministic stand-in for the real ASR engine.

It records how the API called it (so tests can assert language handling and
audio duration) and returns fixed text, which keeps the default test run fast
and free of model downloads.
"""

from __future__ import annotations

import time

import numpy as np

from app.asr.base import TranscriptionResult, TranscriptionSegment

DEFAULT_ZH_TEXT = "你好，欢迎使用语音识别系统。"
DEFAULT_EN_TEXT = "Hello and welcome to the speech recognition system."


class FakeEngine:
    name = "fake-whisper"

    def __init__(
        self,
        *,
        text_zh: str = DEFAULT_ZH_TEXT,
        text_en: str = DEFAULT_EN_TEXT,
        detected_language: str = "zh",
        language_probability: float = 0.99,
        error: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.text_zh = text_zh
        self.text_en = text_en
        self.detected_language = detected_language
        self.language_probability = language_probability
        self.error = error
        self.delay_seconds = delay_seconds

        self.calls: list[dict[str, object]] = []
        self.languages: list[str | None] = []

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error

        duration = round(float(np.asarray(samples).size) / sample_rate, 3)
        self.calls.append({"duration_seconds": duration, "sample_rate": sample_rate, "language": language})
        self.languages.append(language)

        effective = language or self.detected_language
        text = self.text_zh if effective == "zh" else self.text_en
        segments = (TranscriptionSegment(start=0.0, end=duration, text=text),)

        return TranscriptionResult(
            text=text,
            language=effective,
            detected_language=effective,
            language_probability=self.language_probability,
            segments=segments,
        )

    def status(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": "fake-whisper/test",
            "device": "cpu",
            "compute_type": "int8",
            "beam_size": 1,
            "offline": True,
            "loaded": True,
            "load_seconds": 0.0,
            "last_error": None,
        }

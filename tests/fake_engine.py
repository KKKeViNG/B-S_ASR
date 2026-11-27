"""Deterministic stand-in for the real ASR engine.

It records how the API called it (so tests can assert language handling, model
switching and audio duration) and returns fixed text, which keeps the default
test run fast and free of model downloads.
"""

from __future__ import annotations

import time

import numpy as np

from app.asr.base import TranscriptionResult, TranscriptionSegment, Word

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
        model_id: str = "small",
        downloaded: bool = True,
        load_error: Exception | None = None,
        word_timestamps: bool = True,
    ) -> None:
        self.text_zh = text_zh
        self.text_en = text_en
        self.detected_language = detected_language
        self.language_probability = language_probability
        self.error = error
        self.delay_seconds = delay_seconds
        self.model_id = model_id
        self.downloaded = downloaded
        self.load_error = load_error
        self.word_timestamps = word_timestamps
        self.loaded = True

        self.calls: list[dict[str, object]] = []
        self.languages: list[str | None] = []
        self.model_history: list[str] = []
        self.deadlines: list[float | None] = []
        self.word_timestamp_requests: list[bool | None] = []

    # ---- model management ----------------------------------------------

    def set_model(self, model_id: str) -> None:
        if model_id != self.model_id:
            self.model_history.append(model_id)
            self.model_id = model_id
            self.loaded = False

    def load(self):
        if self.load_error is not None:
            raise self.load_error
        self.loaded = True
        return self

    def availability(self) -> dict[str, object]:
        return {
            "available": True,
            "downloaded": self.downloaded,
            "reason": "loaded" if self.loaded else ("cached" if self.downloaded else "will_download"),
        }

    # ---- inference ------------------------------------------------------

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
        *,
        deadline: float | None = None,
        word_timestamps: bool | None = None,
    ) -> TranscriptionResult:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error

        duration = round(float(np.asarray(samples).size) / sample_rate, 3)
        effective = language or self.detected_language
        self.calls.append({"duration_seconds": duration, "sample_rate": sample_rate, "language": language})
        self.languages.append(language)
        self.deadlines.append(deadline)
        self.word_timestamp_requests.append(word_timestamps)

        if deadline is not None and time.perf_counter() >= deadline:
            return TranscriptionResult(
                text="",
                language=effective,
                detected_language=effective,
                segments=(),
                model=self.model_id,
                truncated=True,
            )

        text = self.text_zh if effective == "zh" else self.text_en
        wants_words = self.word_timestamps if word_timestamps is None else bool(word_timestamps)
        segment = TranscriptionSegment(
            start=0.0,
            end=duration,
            text=text,
            words=self._words(text, duration) if wants_words else (),
        )

        return TranscriptionResult(
            text=text,
            language=effective,
            detected_language=effective,
            language_probability=self.language_probability,
            segments=(segment,),
            model=self.model_id,
        )

    def _words(self, text: str, duration: float) -> tuple[Word, ...]:
        if not self.word_timestamps:
            return ()
        tokens = text.split()
        if not tokens:
            return ()
        step = duration / len(tokens)
        return tuple(
            Word(
                word=token,
                start=round(index * step, 3),
                end=round((index + 1) * step, 3),
                probability=0.9,
            )
            for index, token in enumerate(tokens)
        )

    def status(self) -> dict[str, object]:
        availability = self.availability()
        return {
            "name": self.name,
            "model": f"fake-whisper/{self.model_id}",
            "model_id": self.model_id,
            "device": "cpu",
            "compute_type": "int8",
            "beam_size": 1,
            "offline": True,
            "word_timestamps": self.word_timestamps,
            "loaded": self.loaded,
            "load_seconds": 0.0,
            "last_error": None,
            "downloaded": availability["downloaded"],
            "available": availability["available"],
            "reason": availability["reason"],
        }

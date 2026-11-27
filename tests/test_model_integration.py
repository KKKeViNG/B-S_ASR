"""Model integration tests that never download weights.

These cover the contract between the HTTP layer and the engine (lazy loading,
argument passing, result mapping, failure translation) using stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import faster_whisper

from app.asr.base import normalize_text
from app.asr.whisper_engine import WhisperEngine
from app.config import Settings
from app.errors import ModelUnavailableError, TranscriptionFailedError
from tests.helpers import tone


class _StubSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class _StubInfo:
    language = "en"
    language_probability = 0.8734


class _StubModel:
    def __init__(self, segments, info=None, error: Exception | None = None) -> None:
        self._segments = segments
        self._info = info or _StubInfo()
        self._error = error
        self.kwargs: dict[str, object] = {}

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        if self._error is not None:
            raise self._error
        return iter(self._segments), self._info


def test_engine_does_not_load_model_on_construction(tmp_path: Path):
    engine = WhisperEngine(Settings(model_dir=tmp_path / "models"))

    assert engine.is_loaded is False
    status = engine.status()
    assert status["loaded"] is False
    assert status["load_seconds"] is None
    assert status["model"] == "faster-whisper/small"
    assert status["device"] == "cpu"
    assert status["compute_type"] == "int8"


def test_load_failure_becomes_model_unavailable(tmp_path: Path, monkeypatch):
    def broken_model(*args, **kwargs):
        raise OSError("no such file: model.bin")

    monkeypatch.setattr(faster_whisper, "WhisperModel", broken_model)

    engine = WhisperEngine(Settings(model_dir=tmp_path / "models"))
    with pytest.raises(ModelUnavailableError) as excinfo:
        engine.load()

    assert excinfo.value.status_code == 503
    assert "download_model.py" in excinfo.value.detail
    assert "model.bin" in excinfo.value.detail
    assert engine.status()["last_error"] == "no such file: model.bin"


def test_engine_passes_language_and_decoding_options(tmp_path: Path, settings: Settings):
    engine = WhisperEngine(settings)
    stub = _StubModel([_StubSegment(0.0, 0.5, " Hello"), _StubSegment(0.5, 1.0, "  world  ")])
    engine._model = stub

    result = engine.transcribe(tone(1.0), 16000, "en")

    assert stub.kwargs["language"] == "en"
    assert stub.kwargs["task"] == "transcribe"
    assert stub.kwargs["beam_size"] == settings.beam_size
    assert stub.kwargs["condition_on_previous_text"] is False
    assert stub.kwargs["word_timestamps"] is False
    assert result.text == "Hello world", "whitespace must be normalised across segments"
    assert result.language == "en"
    assert result.detected_language == "en"
    assert result.language_probability == 0.8734
    assert [segment.text for segment in result.segments] == [" Hello", "  world  "]


def test_engine_reports_detected_language_in_auto_mode(tmp_path: Path):
    engine = WhisperEngine(Settings(model_dir=tmp_path / "models"))
    engine._model = _StubModel([_StubSegment(0.0, 1.0, " hello there")])

    result = engine.transcribe(tone(1.0), 16000, None)

    assert result.language == "en"
    assert result.detected_language == "en"


def test_engine_rejects_unexpected_sample_rate(tmp_path: Path):
    engine = WhisperEngine(Settings(model_dir=tmp_path / "models"))

    with pytest.raises(TranscriptionFailedError) as excinfo:
        engine.transcribe(tone(1.0, sample_rate=8000), 8000, "zh")

    assert "16000" in excinfo.value.detail
    assert engine.is_loaded is False


def test_engine_rejects_empty_audio(tmp_path: Path):
    engine = WhisperEngine(Settings(model_dir=tmp_path / "models"))

    with pytest.raises(TranscriptionFailedError):
        engine.transcribe(np.zeros(0, dtype=np.float32), 16000, "zh")

    assert engine.is_loaded is False


def test_inference_failure_is_translated(tmp_path: Path):
    engine = WhisperEngine(Settings(model_dir=tmp_path / "models"))
    engine._model = _StubModel([], error=ValueError("decoder exploded"))

    with pytest.raises(TranscriptionFailedError) as excinfo:
        engine.transcribe(tone(1.0), 16000, "zh")

    assert "decoder exploded" in excinfo.value.detail


def test_whitespace_normalisation():
    assert normalize_text("  hello   world \n") == "hello world"
    assert normalize_text("") == ""


def test_engine_serialises_concurrent_inference(tmp_path: Path):
    """Two threads must not use the CTranslate2 model at the same time."""

    import threading
    import time

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    class _SlowModel:
        def transcribe(self, audio, **kwargs):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1
            return iter([_StubSegment(0.0, 1.0, " ok")]), _StubInfo()

    engine = WhisperEngine(Settings(model_dir=tmp_path / "models"))
    engine._model = _SlowModel()

    threads = [threading.Thread(target=engine.transcribe, args=(tone(1.0), 16000, "en")) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["max_active"] == 1

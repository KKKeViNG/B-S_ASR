"""Verification with the **real** local faster-whisper model.

These tests are skipped by default (see ``tests/conftest.py``) because they load
the actual weights and take a few seconds per clip. Run them with::

    python -m pytest tests/test_real_model.py --run-real-model -q

They are the required "real model" counterpart to the mocked tests and use the
bundled samples plus their human reference transcripts.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.asr.whisper_engine import WhisperEngine
from app.config import PROJECT_ROOT, Settings
from app.main import create_app
from tests.helpers import character_error_rate, load_reference_texts, word_error_rate

pytestmark = pytest.mark.real_model

SAMPLES_DIR = PROJECT_ROOT / "samples"


@pytest.fixture(scope="module")
def real_settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="module")
def real_engine(real_settings: Settings) -> WhisperEngine:
    engine = WhisperEngine(real_settings)
    engine.load()
    return engine


@pytest.fixture(scope="module")
def real_client(real_settings: Settings, real_engine: WhisperEngine) -> TestClient:
    app = create_app(settings=real_settings, engine=real_engine)
    with TestClient(app) as client:
        yield client


def _transcribe(client: TestClient, filename: str, language: str) -> dict:
    audio = (SAMPLES_DIR / filename).read_bytes()
    response = client.post(
        "/api/transcribe",
        files={"file": (filename, audio, "audio/wav")},
        data={"language": language},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_model_is_loaded_and_reported(real_engine: WhisperEngine):
    status = real_engine.status()

    assert status["loaded"] is True
    assert status["model"] == "faster-whisper/small"
    assert status["device"] == "cpu"
    assert status["load_seconds"] is not None


def test_chinese_sample_end_to_end(real_client: TestClient):
    references = load_reference_texts()
    payload = _transcribe(real_client, "chinese.wav", "zh")
    print("chinese response:", json.dumps(payload, ensure_ascii=False))

    cer = character_error_rate(references["chinese.wav"], payload["text"])
    print(f"chinese CER = {cer:.4f}")

    assert payload["text"].strip()
    assert payload["language"] == "zh"
    assert payload["detected_language"] == "zh"
    assert 5.0 <= payload["duration_seconds"] <= 15.0
    assert payload["processing_seconds"] > 0
    assert cer <= 0.20, f"Chinese CER too high: {cer:.3f}"


def test_english_sample_end_to_end(real_client: TestClient):
    references = load_reference_texts()
    payload = _transcribe(real_client, "english.wav", "en")
    print("english response:", json.dumps(payload, ensure_ascii=False))

    wer = word_error_rate(references["english.wav"], payload["text"])
    print(f"english WER = {wer:.4f}")

    assert payload["text"].strip()
    assert payload["language"] == "en"
    assert payload["detected_language"] == "en"
    assert 5.0 <= payload["duration_seconds"] <= 15.0
    assert wer <= 0.25, f"English WER too high: {wer:.3f}"


def test_auto_language_detection_with_real_model(real_client: TestClient):
    chinese = _transcribe(real_client, "chinese.wav", "auto")
    english = _transcribe(real_client, "english.wav", "auto")

    assert chinese["language"] == "zh"
    assert english["language"] == "en"
    assert chinese["detected_language"] == "zh"
    assert english["detected_language"] == "en"


def test_repeated_real_requests_are_stable(real_client: TestClient):
    first = _transcribe(real_client, "english.wav", "en")
    second = _transcribe(real_client, "english.wav", "en")

    assert first["text"] == second["text"]

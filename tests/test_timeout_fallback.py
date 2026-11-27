"""Deadline handling, partial results, model fallback and error metadata."""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from app.errors import TranscriptionFailedError
from app.main import create_app
from tests.fake_engine import FakeEngine
from tests.helpers import post_audio, wav_bytes


def test_timeout_for_scales_with_audio(settings):
    assert settings.timeout_for(0) == 15.0
    assert settings.timeout_for(10) == 55.0
    assert settings.timeout_for(10_000) == 600.0

    overridden = replace(settings, request_timeout_seconds=30.0)
    assert overridden.timeout_for(100) == 30.0


def test_deadline_exceeded_returns_504_with_metadata(settings):
    fast_timeout = replace(settings, request_timeout_seconds=0.05)
    engine = FakeEngine(delay_seconds=0.2)
    app = create_app(settings=fast_timeout, engine=engine)

    with TestClient(app) as client:
        response = post_audio(client, wav_bytes(1.0), language="zh")

    assert response.status_code == 504
    payload = response.json()
    assert payload["code"] == "timeout"
    assert payload["stage"] == "inference"
    assert payload["retryable"] is True


class TruncatingEngine(FakeEngine):
    """Returns text but flags the result as cut off by the deadline."""

    def transcribe(self, samples, sample_rate, language, *, deadline=None, word_timestamps=None):
        result = super().transcribe(
            samples, sample_rate, language, word_timestamps=word_timestamps
        )
        return replace(result, truncated=True)


def test_partial_text_is_returned_with_warning(settings):
    app = create_app(settings=settings, engine=TruncatingEngine())

    with TestClient(app) as client:
        response = post_audio(client, wav_bytes(1.0), language="zh")

    assert response.status_code == 200
    payload = response.json()
    assert payload["truncated"] is True
    assert "deadline_exceeded" in payload["warning"]
    assert payload["text"]


class FlakyEngine(FakeEngine):
    """Fails on the primary model, succeeds on the fallback."""

    def transcribe(self, samples, sample_rate, language, *, deadline=None, word_timestamps=None):
        if self.model_id == "small":
            raise TranscriptionFailedError("simulated decoder failure on small")
        return super().transcribe(
            samples, sample_rate, language, word_timestamps=word_timestamps
        )


def test_fallback_model_is_used_and_restored(settings):
    fallback_settings = replace(settings, fallback_model="base")
    engine = FlakyEngine()
    app = create_app(settings=fallback_settings, engine=engine)

    with TestClient(app) as client:
        response = post_audio(client, wav_bytes(1.0), language="zh")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "faster-whisper/base"
    assert "degraded_model" in payload["warning"]
    assert payload["text"]
    assert engine.model_id == "small", "the primary model must be restored"

    # A second request must try the primary model again and degrade again.
    with TestClient(app) as client:
        second = post_audio(client, wav_bytes(1.0), language="zh")
    assert second.status_code == 200
    assert second.json()["model"] == "faster-whisper/base"


def test_fallback_disabled_keeps_the_original_error(settings):
    engine = FlakyEngine()
    app = create_app(settings=settings, engine=engine)

    with TestClient(app) as client:
        response = post_audio(client, wav_bytes(1.0), language="zh")

    assert response.status_code == 500
    assert response.json()["code"] == "transcription_failed"
    assert engine.model_id == "small"


def test_error_payloads_carry_stage_and_retryable(client):
    unsupported = post_audio(client, b"not audio", filename="notes.txt")
    assert unsupported.status_code == 415
    assert unsupported.json()["stage"] == "request"
    assert unsupported.json()["retryable"] is False

    corrupt = post_audio(client, wav_bytes(1.0)[:60], language="zh")
    assert corrupt.status_code == 422
    assert corrupt.json()["stage"] == "decode"

    bad_language = post_audio(client, wav_bytes(1.0), language="de")
    assert bad_language.status_code == 422
    assert bad_language.json()["stage"] == "request"


def test_words_are_returned_in_segments(client):
    payload = post_audio(client, wav_bytes(1.0), language="en").json()

    segment = payload["segments"][0]
    assert segment["words"], "word timestamps should be present"
    assert all(word["start"] <= word["end"] for word in segment["words"])
    assert payload["truncated"] is False

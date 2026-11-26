"""Error handling: bad input, oversize payloads, model failures."""

from __future__ import annotations

import pytest

from app.errors import ModelUnavailableError, TranscriptionFailedError
from tests.helpers import post_audio, wav_bytes


def test_missing_file_field_returns_validation_error(client):
    response = client.post("/api/transcribe", data={"language": "zh"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "validation_error"
    assert "file" in payload["detail"]


def test_empty_file_is_rejected(client, fake_engine):
    response = post_audio(client, b"", language="zh")

    assert response.status_code == 400
    assert response.json()["code"] == "bad_request"
    assert "empty" in response.json()["detail"].lower()
    assert fake_engine.calls == []


@pytest.mark.parametrize("filename", ["notes.txt", "document.pdf", "archive.zip"])
def test_unsupported_format_returns_415(client, filename):
    response = post_audio(client, b"whatever bytes", filename=filename)

    assert response.status_code == 415
    payload = response.json()
    assert payload["code"] == "unsupported_format"
    assert ".wav" in payload["detail"]


def test_file_without_extension_is_rejected(client):
    response = post_audio(client, wav_bytes(0.5), filename="recording")

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_format"


def test_corrupted_wav_is_rejected(client, fake_engine):
    response = post_audio(client, wav_bytes(1.0)[:60], language="zh")

    assert response.status_code == 422
    assert response.json()["code"] == "audio_decode_error"
    assert fake_engine.calls == []


def test_upload_larger_than_limit_returns_413(client, settings):
    oversized = b"\x00" * (settings.max_upload_bytes + 1)

    response = post_audio(client, oversized, language="zh")

    assert response.status_code == 413
    payload = response.json()
    assert payload["code"] == "payload_too_large"
    assert "MB" in payload["detail"] or "bytes" in payload["detail"]


def test_audio_longer_than_limit_returns_413(client, settings):
    response = post_audio(client, wav_bytes(settings.max_audio_seconds + 2), language="zh")

    assert response.status_code == 413
    assert response.json()["code"] == "payload_too_large"
    assert "longer than" in response.json()["detail"]


def test_model_unavailable_returns_503(client, fake_engine):
    fake_engine.error = ModelUnavailableError(
        "The local speech recognition model could not be loaded (model: faster-whisper/small)."
    )

    response = post_audio(client, wav_bytes(1.0), language="zh")

    assert response.status_code == 503
    payload = response.json()
    assert payload["code"] == "model_unavailable"
    assert "model" in payload["detail"].lower()


def test_inference_failure_returns_500(client, fake_engine):
    fake_engine.error = TranscriptionFailedError("Speech recognition failed while processing this audio: boom")

    response = post_audio(client, wav_bytes(1.0), language="zh")

    assert response.status_code == 500
    assert response.json()["code"] == "transcription_failed"


def test_unexpected_exception_returns_500_json(raw_client, fake_engine):
    fake_engine.error = RuntimeError("unexpected internal failure")

    response = post_audio(raw_client, wav_bytes(1.0), language="zh")

    assert response.status_code == 500
    payload = response.json()
    assert payload["code"] == "internal_error"
    assert payload["detail"]


def test_error_responses_are_always_json_with_code_and_detail(client):
    for audio, filename, language in [
        (b"", "clip.wav", "zh"),
        (b"nope", "clip.txt", "zh"),
        (wav_bytes(0.5), "clip.wav", "de"),
    ]:
        response = post_audio(client, audio, filename=filename, language=language)
        payload = response.json()
        assert response.status_code >= 400
        assert isinstance(payload["code"], str) and payload["code"]
        assert isinstance(payload["detail"], str) and payload["detail"]


def test_service_stays_healthy_after_rejected_requests(client, fake_engine):
    assert post_audio(client, b"", language="zh").status_code == 400
    assert post_audio(client, b"nope", filename="clip.txt", language="zh").status_code == 415

    response = post_audio(client, wav_bytes(1.0), language="zh")

    assert response.status_code == 200
    assert response.json()["text"] == fake_engine.text_zh
    assert client.get("/health").json() == {"status": "ok"}

"""Tests for the required `POST /api/transcribe` contract."""

from __future__ import annotations

import pytest

from tests.helpers import post_audio, silent_wav_bytes, wav_bytes

REQUIRED_KEYS = {"text", "language", "duration_seconds", "processing_seconds"}


def test_transcribe_chinese_clip(client, fake_engine):
    response = post_audio(client, wav_bytes(2.0), language="zh")

    assert response.status_code == 200
    payload = response.json()

    assert REQUIRED_KEYS.issubset(payload.keys())
    assert payload["text"] == fake_engine.text_zh
    assert payload["language"] == "zh"
    assert payload["duration_seconds"] == pytest.approx(2.0, abs=0.05)
    assert payload["processing_seconds"] >= 0
    assert payload["model"] == "faster-whisper/small"
    assert payload["segments"]

    assert fake_engine.languages == ["zh"]
    assert fake_engine.calls[0]["duration_seconds"] == pytest.approx(2.0, abs=0.05)


def test_transcribe_english_clip(client, fake_engine):
    response = post_audio(client, wav_bytes(1.5), language="en")

    assert response.status_code == 200
    payload = response.json()

    assert payload["text"] == fake_engine.text_en
    assert payload["language"] == "en"
    assert fake_engine.languages == ["en"]


def test_transcribe_auto_detect_returns_detected_language(client, fake_engine):
    response = post_audio(client, wav_bytes(1.5), language="auto")

    assert response.status_code == 200
    payload = response.json()

    # The fake engine always detects Chinese unless it is told otherwise.
    assert payload["language"] == "zh"
    assert payload["detected_language"] == "zh"
    assert fake_engine.languages == [None], "auto mode must not force a language"


def test_transcribe_without_language_uses_server_default(client, fake_engine, settings):
    response = post_audio(client, wav_bytes(1.0), language=None)

    assert response.status_code == 200
    assert response.json()["language"] == settings.default_language
    assert fake_engine.languages == [settings.default_language]


def test_duration_reflects_real_audio_length(client):
    response = post_audio(client, wav_bytes(3.0), language="zh")

    assert response.status_code == 200
    assert response.json()["duration_seconds"] == pytest.approx(3.0, abs=0.05)


def test_processing_seconds_is_measured_by_the_server(client):
    response = post_audio(client, wav_bytes(1.0), language="zh")

    assert response.status_code == 200
    assert 0 <= response.json()["processing_seconds"] < 5


def test_multiple_requests_are_independent(client, fake_engine):
    first = post_audio(client, wav_bytes(1.0), language="zh")
    second = post_audio(client, wav_bytes(2.0), language="en")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["text"] != second.json()["text"]
    assert fake_engine.languages == ["zh", "en"]


def test_silent_clip_returns_empty_text_without_calling_the_model(client, fake_engine):
    response = post_audio(client, silent_wav_bytes(2.0), language="zh")

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == ""
    assert payload["warning"] == "no_speech_detected"
    assert payload["language"] == "zh"
    assert fake_engine.calls == []


def test_upload_of_stereo_44k_file_works(client):
    from tests.helpers import stereo_wav_bytes

    response = post_audio(client, stereo_wav_bytes(1.0, 44100), language="zh")

    assert response.status_code == 200
    assert response.json()["duration_seconds"] == pytest.approx(1.0, abs=0.06)

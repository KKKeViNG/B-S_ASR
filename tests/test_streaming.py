"""WebSocket streaming tests (bonus feature)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.errors import ModelUnavailableError
from app.main import create_app
from tests.helpers import silent_wav_bytes, wav_bytes
from tests.fake_engine import FakeEngine


def _start(websocket, language: str = "zh") -> dict:
    websocket.send_json({"type": "start", "language": language})
    return websocket.receive_json()


def test_streaming_ready_partial_and_final(client, fake_engine):
    with client.websocket_connect("/ws/stream") as websocket:
        ready = _start(websocket, "zh")
        assert ready["type"] == "ready"
        assert ready["language"] == "zh"
        assert ready["chunk_seconds"] == pytest.approx(2.5)
        assert ready["model"] == "faster-whisper/small"

        websocket.send_bytes(wav_bytes(2.0))
        first = websocket.receive_json()
        assert first["type"] == "partial"
        assert first["chunk_index"] == 1
        assert first["text"] == fake_engine.text_zh
        assert first["duration_seconds"] == pytest.approx(2.0, abs=0.05)

        websocket.send_bytes(wav_bytes(1.0))
        second = websocket.receive_json()
        assert second["type"] == "partial"
        assert second["chunk_index"] == 2
        assert second["text"].count(fake_engine.text_zh) == 2
        assert second["duration_seconds"] == pytest.approx(3.0, abs=0.05)

        websocket.send_json({"type": "stop"})
        final = websocket.receive_json()

    assert final["type"] == "final"
    assert final["text"].count(fake_engine.text_zh) == 2
    assert final["chunks"] == 2
    assert final["duration_seconds"] == pytest.approx(3.0, abs=0.05)
    assert final["processing_seconds"] >= 0
    assert final["language"] == "zh"


def test_silent_chunk_is_not_sent_to_the_model(client, fake_engine):
    with client.websocket_connect("/ws/stream") as websocket:
        _start(websocket)
        websocket.send_bytes(silent_wav_bytes(1.5))
        partial = websocket.receive_json()
        websocket.send_json({"type": "stop"})
        final = websocket.receive_json()

    assert partial["type"] == "partial"
    assert partial["silent"] is True
    assert final["text"] == ""
    assert fake_engine.calls == []


def test_chunk_too_short_is_skipped(client, fake_engine):
    with client.websocket_connect("/ws/stream") as websocket:
        _start(websocket)
        websocket.send_bytes(wav_bytes(0.1))
        partial = websocket.receive_json()

    assert partial["type"] == "partial"
    assert partial["skipped"] == "chunk_too_short"
    assert fake_engine.calls == []


def test_invalid_start_language_is_reported_and_session_continues(client, fake_engine):
    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_json({"type": "start", "language": "de"})
        error = websocket.receive_json()

        assert error["type"] == "error"
        assert error["code"] == "invalid_language"

        ready = _start(websocket, "en")
        assert ready["type"] == "ready"
        assert ready["language"] == "en"

        websocket.send_bytes(wav_bytes(1.5))
        partial = websocket.receive_json()
        assert partial["text"] == fake_engine.text_en


def test_malformed_control_frame_returns_error(client):
    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_text("not json")
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "bad_request"


def test_unknown_control_message_returns_error(client):
    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_json({"type": "rewind"})
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "bad_request"


def test_oversized_chunk_is_rejected(client, settings):
    payload = b"\x00" * (settings.max_stream_chunk_bytes + 1)

    with client.websocket_connect("/ws/stream") as websocket:
        _start(websocket)
        websocket.send_bytes(payload)
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "payload_too_large"


def test_undecodable_chunk_reports_error_and_keeps_session(client, fake_engine):
    with client.websocket_connect("/ws/stream") as websocket:
        _start(websocket)
        websocket.send_bytes(b"not-an-audio-file" * 100)
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "audio_decode_error"

        websocket.send_bytes(wav_bytes(1.5))
        partial = websocket.receive_json()

    assert partial["type"] == "partial"
    assert partial["text"] == fake_engine.text_zh


def test_fatal_model_error_stops_the_session(client, fake_engine):
    fake_engine.error = ModelUnavailableError("model weights missing")

    with client.websocket_connect("/ws/stream") as websocket:
        _start(websocket)
        websocket.send_bytes(wav_bytes(1.5))
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "model_unavailable"
    assert error["fatal"] is True


def test_auto_language_streaming_uses_detection(client, fake_engine):
    with client.websocket_connect("/ws/stream") as websocket:
        _start(websocket, "auto")
        websocket.send_bytes(wav_bytes(1.5))
        partial = websocket.receive_json()
        websocket.send_json({"type": "stop"})
        final = websocket.receive_json()

    assert partial["type"] == "partial"
    assert final["type"] == "final"
    assert final["language"] == "zh"
    assert fake_engine.languages == [None]


def test_streaming_can_be_disabled():
    settings = Settings(streaming_enabled=False)
    app = create_app(settings=settings, engine=FakeEngine())

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/stream") as websocket:
                websocket.receive_json()


def test_refine_returns_a_clean_final_transcript(client, fake_engine):
    """The browser re-sends the full clip; the final text must come from it."""

    with client.websocket_connect("/ws/stream") as websocket:
        _start(websocket)
        websocket.send_bytes(wav_bytes(2.0))
        websocket.receive_json()

        websocket.send_json({"type": "stop"})
        assembled = websocket.receive_json()
        assert assembled["type"] == "final"
        assert assembled["refined"] is False

        websocket.send_json({"type": "refine"})
        websocket.send_bytes(wav_bytes(2.5))
        refined = websocket.receive_json()

    assert refined["type"] == "final"
    assert refined["refined"] is True
    assert refined["text"] == fake_engine.text_zh
    assert refined["duration_seconds"] == pytest.approx(2.5, abs=0.05)
    assert refined["language"] == "zh"
    assert refined["processing_seconds"] >= 0
    assert refined["segments"][0]["words"], "refined final should expose word timings"


def test_refine_with_undecodable_audio_reports_refine_error(client):
    with client.websocket_connect("/ws/stream") as websocket:
        _start(websocket)
        websocket.send_bytes(wav_bytes(1.0))
        websocket.receive_json()
        websocket.send_json({"type": "stop"})
        websocket.receive_json()

        websocket.send_json({"type": "refine"})
        websocket.send_bytes(b"definitely-not-audio" * 50)
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert error["code"] == "audio_decode_error"
    assert error["refine"] is True


def test_ping_returns_pong(client):
    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"

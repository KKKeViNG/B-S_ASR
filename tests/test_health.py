"""HTTP health check, page serving, configuration and static asset tests."""

from __future__ import annotations

import pytest


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/docs", "/openapi.json"])
def test_api_documentation_available(client, path):
    assert client.get(path).status_code == 200


def test_root_serves_web_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "本地语音识别" in body
    assert "/static/js/app.js" in body
    assert "/static/css/app.css" in body


def test_config_endpoint_exposes_limits_and_model(client, settings):
    payload = client.get("/api/config").json()

    assert payload["languages"] == ["zh", "en", "auto"]
    assert payload["max_upload_bytes"] == settings.max_upload_bytes
    assert payload["max_recording_seconds"] == settings.max_recording_seconds
    assert ".wav" in payload["allowed_extensions"]
    assert payload["model"]["name"] == "faster-whisper/small"
    assert payload["model"]["device"] == "cpu"
    assert payload["streaming"]["endpoint"] == "/ws/stream"
    assert payload["streaming"]["enabled"] is True


def test_model_status_endpoint(client):
    payload = client.get("/api/model").json()

    assert payload["name"] == "fake-whisper"
    assert payload["supported_languages"] == ["zh", "en", "auto"]
    assert payload["loaded"] is True


@pytest.mark.parametrize(
    "path",
    [
        "/static/css/app.css",
        "/static/js/app.js",
        "/static/js/api.js",
        "/static/js/recorder.js",
        "/static/js/streaming.js",
        "/static/js/wav.js",
        "/static/js/pcm-worklet.js",
        "/samples/chinese.wav",
        "/samples/english.wav",
    ],
)
def test_static_assets_are_served(client, path):
    response = client.get(path)

    assert response.status_code == 200
    assert response.content


def test_unknown_path_returns_json_error(client):
    response = client.get("/api/does-not-exist")

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "http_error"
    assert payload["detail"]

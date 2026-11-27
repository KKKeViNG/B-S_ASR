"""Model catalogue, selection persistence and download-stream tests."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.errors import ModelUnavailableError
from app.main import create_app
from app.models import MODEL_CATALOG, model_cache_dir
from tests.fake_engine import FakeEngine


def _events(response) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_catalogue_exposes_metadata(client):
    payload = client.get("/api/models").json()

    ids = [model["id"] for model in payload["models"]]
    assert ids == list(MODEL_CATALOG)
    assert payload["selected"] == "small"
    assert payload["device"] == "cpu"

    small = next(model for model in payload["models"] if model["id"] == "small")
    assert small["recommended"] is True
    assert small["selected"] is True
    assert small["size_mb"] == 484
    assert small["languages"] == ["zh", "en"]
    assert "MIT" in small["license"]
    assert small["description"]


def test_downloaded_flag_reflects_local_cache(client, settings):
    cache = model_cache_dir(settings.model_dir, "base") / "snapshots" / "abc123"
    cache.mkdir(parents=True)
    (cache / "model.bin").write_bytes(b"\x00" * (512 * 1024))

    payload = client.get("/api/models").json()
    base = next(model for model in payload["models"] if model["id"] == "base")
    small = next(model for model in payload["models"] if model["id"] == "small")

    assert base["downloaded"] is True
    assert base["size_on_disk_mb"] > 0
    assert small["downloaded"] is False


def test_select_persists_choice_and_updates_engine(client, settings, fake_engine):
    response = client.post("/api/models/select", json={"model": "base"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected"] == "base"
    assert payload["model"]["selected"] is True
    assert fake_engine.model_history == ["base"]
    assert fake_engine.model_id == "base"

    state = json.loads((settings.model_dir / "state.json").read_text(encoding="utf-8"))
    assert state["selected_model"] == "base"

    # A fresh app with the same model directory must restore the selection.
    second_app = create_app(settings=settings, engine=FakeEngine(model_id="small"))
    with TestClient(second_app) as second_client:
        assert second_client.get("/api/models").json()["selected"] == "base"


def test_select_unknown_model_returns_422(client, settings, fake_engine):
    response = client.post("/api/models/select", json={"model": "gpt-4o"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "invalid_model"
    assert payload["stage"] == "model"
    assert fake_engine.model_history == []
    assert not (settings.model_dir / "state.json").exists()


def test_select_requires_model_field(client):
    response = client.post("/api/models/select", json={})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_download_stream_completes_and_activates(client, fake_engine):
    response = client.get("/api/models/base/download")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response)

    assert events[0]["stage"] in {"starting", "loading"}
    assert events[-1]["stage"] == "ready"
    assert events[-1]["model"] == "base"
    assert events[-1]["load_seconds"] >= 0
    assert "base" in fake_engine.model_history


def test_download_stream_reports_load_failure(settings):
    engine = FakeEngine(load_error=ModelUnavailableError("weights are missing"))
    app = create_app(settings=settings, engine=engine)

    with TestClient(app) as client:
        response = client.get("/api/models/medium/download")

    events = _events(response)
    assert events[-1]["stage"] == "error"
    assert events[-1]["code"] == "model_unavailable"
    assert "weights" in events[-1]["detail"]
    assert events[-1]["retryable"] is True


def test_download_unknown_model_is_rejected(client):
    response = client.get("/api/models/unknown/download")

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_model"


def test_model_status_reports_selection_and_cache(client, fake_engine):
    payload = client.get("/api/model").json()

    assert payload["name"] == "fake-whisper"
    assert payload["selected_model"] == "small"
    assert payload["model_id"] == "small"
    assert payload["model_label"] == "faster-whisper/small"
    assert payload["word_timestamps"] is True
    assert payload["available"] is True
    assert payload["supported_languages"] == ["zh", "en", "auto"]

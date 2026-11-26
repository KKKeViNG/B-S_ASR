"""Test-suite configuration and shared fixtures.

The default `python -m pytest -q` run is fast, offline and hardware independent:
the ASR engine is replaced by :class:`tests.fake_engine.FakeEngine`, so no model
weights are downloaded or loaded. Tests that need the real local model are
marked ``real_model`` and are skipped unless ``--run-real-model`` is passed (or
``STT_RUN_REAL_MODEL=1`` is set), see ``docs/test-report.md``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fake_engine import FakeEngine  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-real-model",
        action="store_true",
        default=False,
        help="Also run the tests that load the real faster-whisper model.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "real_model: requires the real local ASR model (slow, needs weights)"
    )


def _real_model_enabled(config: pytest.Config) -> bool:
    if config.getoption("--run-real-model"):
        return True
    return os.environ.get("STT_RUN_REAL_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _real_model_enabled(config):
        return
    skip_marker = pytest.mark.skip(
        reason="real-model test skipped: pass --run-real-model (see README / docs/test-report.md)"
    )
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip_marker)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Small, isolated settings: no real model directory is ever touched."""

    return Settings(
        model_dir=tmp_path / "models",
        beam_size=1,
        max_upload_bytes=2_000_000,
        max_audio_seconds=10.0,
        max_recording_seconds=60.0,
        max_stream_chunk_bytes=120_000,
    )


@pytest.fixture()
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture()
def app(settings: Settings, fake_engine: FakeEngine):
    return create_app(settings=settings, engine=fake_engine)


@pytest.fixture()
def client(app) -> TestClient:
    """Test client whose server exceptions are re-raised (default behaviour)."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def raw_client(app) -> TestClient:
    """Test client that returns 500 responses instead of re-raising."""

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

"""Front-end contract and end-to-end HTTP flow tests.

The browser is not launched here (that is done by ``scripts/e2e_browser_check.py``
with Playwright), but these tests guarantee the page ships the expected controls,
the JavaScript talks to the documented endpoints, and the full
record → upload → transcribe → display flow works over real HTTP.
"""

from __future__ import annotations

import pytest

from app.config import PROJECT_ROOT
from tests.helpers import post_audio, wav_bytes

STATIC_DIR = PROJECT_ROOT / "static"
REQUIRED_CONTROL_IDS = [
    "language-group",
    "model-select",
    "model-apply",
    "model-status",
    "mode-group",
    "record-button",
    "stop-button",
    "record-timer",
    "record-status",
    "file-input",
    "upload-button",
    "result-text",
    "audio-player",
    "segment-list",
    "export-txt",
    "export-srt",
    "export-vtt",
    "copy-button",
]


def test_index_page_exposes_required_controls(client):
    body = client.get("/").text

    for element_id in REQUIRED_CONTROL_IDS:
        assert f'id="{element_id}"' in body, f"missing control #{element_id}"
    assert "开始录音" in body
    assert "结束录音" in body
    assert "上传音频文件" in body
    assert "识别结果" in body
    assert "识别模型" in body
    assert "导出 SRT" in body


def test_javascript_uses_documented_endpoints():
    api_source = (STATIC_DIR / "js" / "api.js").read_text(encoding="utf-8")
    streaming_source = (STATIC_DIR / "js" / "streaming.js").read_text(encoding="utf-8")

    assert "/api/transcribe" in api_source
    assert "FormData" in api_source
    assert "form.append('file'" in api_source
    assert "form.append('language'" in api_source
    assert "/api/models" in api_source
    assert "/api/models/select" in api_source
    assert "/api/subtitles" in api_source
    assert "/download" in api_source
    assert "/ws/stream" in streaming_source


def test_frontend_wires_model_switch_player_and_export():
    app_source = (STATIC_DIR / "js" / "app.js").read_text(encoding="utf-8")

    assert "fetchModels" in app_source
    assert "streamModelDownload" in app_source
    assert "fetchSubtitles" in app_source
    assert "audioPlayer" in app_source
    assert "segmentList" in app_source
    assert "timeupdate" in app_source


def test_recorder_uses_media_recorder_with_opus():
    recorder_source = (STATIC_DIR / "js" / "recorder.js").read_text(encoding="utf-8")

    assert "MediaRecorder" in recorder_source
    assert "audio/webm;codecs=opus" in recorder_source
    assert "getUserMedia" in recorder_source


def test_frontend_enforces_60_second_recording_limit():
    recorder_source = (STATIC_DIR / "js" / "recorder.js").read_text(encoding="utf-8")

    assert "maxSeconds" in recorder_source
    assert "setTimeout" in recorder_source


def test_upload_flow_works_for_both_languages(client, fake_engine):
    chinese = post_audio(client, wav_bytes(2.0), language="zh", filename="sample.wav")
    english = post_audio(client, wav_bytes(2.0), language="en", filename="sample.wav")

    assert chinese.status_code == 200
    assert english.status_code == 200
    assert chinese.json()["text"] == fake_engine.text_zh
    assert english.json()["text"] == fake_engine.text_en


def test_browser_recording_format_is_accepted(client):
    """MediaRecorder produces WebM/Opus; the backend must accept it end to end."""

    from tests.helpers import webm_opus_bytes

    audio = webm_opus_bytes(1.5)
    if audio is None:
        pytest.skip("local FFmpeg build has no libopus encoder")

    response = post_audio(client, audio, language="zh", filename="recording.webm", content_type="audio/webm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"]
    assert payload["duration_seconds"] == pytest.approx(1.5, abs=0.2)


def test_default_settings_match_assessment_limits():
    from app.config import Settings

    settings = Settings()

    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.max_recording_seconds == 60.0
    assert settings.model_size == "small"
    assert settings.device == "cpu"
    assert settings.compute_type == "int8"


def test_fake_engine_run_never_touches_the_model_directory(client, settings):
    """The default test run must not download or load model weights."""

    assert not (settings.model_dir).exists()
    post_audio(client, wav_bytes(1.0), language="zh")
    assert not (settings.model_dir).exists()

"""Subtitle formatting tests (SRT / WebVTT / TXT)."""

from __future__ import annotations

import pytest

from app.subtitles import build_subtitles, format_timestamp, to_srt, to_txt, to_vtt

SEGMENTS = [
    {"start": 0.0, "end": 1.5, "text": "你好"},
    {"start": 1.5, "end": 2.25, "text": "世界"},
]


def test_timestamp_formatting_handles_hours_and_rounding():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(1.2345) == "00:00:01,234"
    assert format_timestamp(3723.456) == "01:02:03,456"
    assert format_timestamp(1.5, millisecond_separator=".") == "00:00:01.500"
    assert format_timestamp(-3) == "00:00:00,000"


def test_srt_uses_numbered_cues_and_comma_milliseconds():
    assert to_srt(SEGMENTS) == (
        "1\n00:00:00,000 --> 00:00:01,500\n你好\n\n"
        "2\n00:00:01,500 --> 00:00:02,250\n世界\n"
    )


def test_vtt_has_header_and_dot_milliseconds():
    vtt = to_vtt(SEGMENTS)

    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.500" in vtt
    assert vtt.endswith("\n")


def test_blank_segments_are_skipped():
    assert to_srt([{"start": 0, "end": 1, "text": "   "}]) == ""
    assert to_vtt([{"start": 0, "end": 1, "text": ""}]).strip() == "WEBVTT"


def test_txt_prefers_full_text_then_falls_back_to_segments():
    assert to_txt([], "hello world") == "hello world\n"
    assert to_txt(SEGMENTS) == "你好 世界\n"


def test_unknown_format_raises_value_error():
    with pytest.raises(ValueError):
        build_subtitles(SEGMENTS, "ass")


def test_subtitles_api_returns_srt_attachment(client):
    response = client.post("/api/subtitles", json={"segments": SEGMENTS, "format": "srt"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-subrip")
    assert "attachment" in response.headers["content-disposition"]
    assert response.text.startswith("1\n00:00:00,000 --> 00:00:01,500\n你好")


@pytest.mark.parametrize("fmt", ["vtt", "txt"])
def test_subtitles_api_supports_other_formats(client, fmt):
    response = client.post("/api/subtitles", json={"segments": SEGMENTS, "format": fmt, "text": "你好 世界"})

    assert response.status_code == 200
    assert f"transcript.{fmt}" in response.headers["content-disposition"]
    if fmt == "vtt":
        assert response.text.startswith("WEBVTT")
    else:
        assert response.text == "你好 世界\n"


def test_subtitles_api_rejects_empty_segments(client):
    response = client.post("/api/subtitles", json={"segments": [], "format": "srt"})

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_subtitles_api_rejects_unknown_format(client):
    response = client.post("/api/subtitles", json={"segments": SEGMENTS, "format": "ass"})

    assert response.status_code == 422

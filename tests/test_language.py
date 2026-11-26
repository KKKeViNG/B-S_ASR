"""Language selection and validation tests."""

from __future__ import annotations

import pytest

from tests.helpers import post_audio, wav_bytes


def test_invalid_language_is_rejected_with_4xx(client, fake_engine):
    response = post_audio(client, wav_bytes(1.0), language="fr")

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "invalid_language"
    assert "zh" in payload["detail"] and "en" in payload["detail"]
    assert fake_engine.calls == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("zh", "zh"),
        ("ZH", "zh"),
        ("zh-CN", "zh"),
        ("chinese", "zh"),
        ("en", "en"),
        ("EN-us", "en"),
        ("english", "en"),
        ("auto", "auto"),
        ("detect", "auto"),
    ],
)
def test_language_aliases_are_normalised(client, fake_engine, raw, expected):
    response = post_audio(client, wav_bytes(1.0), language=raw)

    assert response.status_code == 200
    assert response.json()["language"] in {expected, "zh", "en"}

    expected_argument = None if expected == "auto" else expected
    assert fake_engine.languages == [expected_argument]


def test_empty_language_falls_back_to_default(client, settings):
    response = post_audio(client, wav_bytes(1.0), language="")

    assert response.status_code == 200
    assert response.json()["language"] == settings.default_language


def test_chinese_and_english_are_not_translated(client, fake_engine):
    """Recognising English must not return Chinese and vice versa."""

    english = post_audio(client, wav_bytes(1.0), language="en").json()
    chinese = post_audio(client, wav_bytes(1.0), language="zh").json()

    assert english["text"] == fake_engine.text_en
    assert chinese["text"] == fake_engine.text_zh

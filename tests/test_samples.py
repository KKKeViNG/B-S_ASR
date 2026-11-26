"""Checks on the bundled sample audio and its reference transcripts."""

from __future__ import annotations

import pytest

from app.audio import decode_audio
from tests.helpers import SAMPLES_DIR, load_reference_texts


@pytest.mark.parametrize("filename", ["chinese.wav", "english.wav"])
def test_sample_exists_and_is_usable(filename):
    path = SAMPLES_DIR / filename

    assert path.is_file(), f"missing bundled sample {path}"
    assert path.stat().st_size > 10_000
    assert path.stat().st_size < 5 * 1024 * 1024

    decoded = decode_audio(path.read_bytes())
    assert decoded.sample_rate == 16000
    assert 5.0 <= decoded.duration_seconds <= 15.0, "assessment requires 5-15 s samples"
    assert not decoded.is_silent


def test_reference_transcripts_cover_both_samples():
    references = load_reference_texts()

    assert set(references) == {"chinese.wav", "english.wav"}
    assert references["chinese.wav"].strip()
    assert references["english.wav"].strip()
    assert any("\u4e00" <= char <= "\u9fff" for char in references["chinese.wav"])
    assert all(ord(char) < 128 for char in references["english.wav"])


def test_samples_are_16k_mono_pcm_wav():
    import wave

    for filename in ("chinese.wav", "english.wav"):
        with wave.open(str(SAMPLES_DIR / filename), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16000

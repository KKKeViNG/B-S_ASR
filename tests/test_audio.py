"""Unit tests for the audio layer: sniffing, validation, decoding, encoding."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio import (
    SILENCE_RMS_THRESHOLD,
    decode_audio,
    detect_container,
    encode_wav,
    validate_upload,
)
from app.config import Settings
from app.errors import AudioDecodeError, BadRequestError, PayloadTooLargeError, UnsupportedFormatError
from tests.helpers import silent_wav_bytes, stereo_wav_bytes, tone, wav_bytes

SETTINGS = Settings()


def test_detect_container_recognises_common_browser_formats():
    assert detect_container(wav_bytes(0.2)) == "wav"
    assert detect_container(b"\x1aE\xdf\xa3" + b"\x00" * 32) == "webm"
    assert detect_container(b"OggS" + b"\x00" * 32) == "ogg"
    assert detect_container(b"fLaC" + b"\x00" * 32) == "flac"
    assert detect_container(b"ID3" + b"\x00" * 32) == "mp3"
    assert detect_container(b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 16) == "mp4"
    assert detect_container(b"not audio at all") is None


def test_decode_wav_reports_duration_and_samples():
    decoded = decode_audio(wav_bytes(seconds=2.0))

    assert decoded.container == "wav"
    assert decoded.sample_rate == 16000
    assert decoded.duration_seconds == pytest.approx(2.0, abs=0.02)
    assert decoded.samples.dtype == np.float32
    assert decoded.samples.size == pytest.approx(32000, abs=64)
    assert not decoded.is_silent


def test_decode_resamples_stereo_44k_to_16k_mono():
    decoded = decode_audio(stereo_wav_bytes(seconds=1.0, sample_rate=44100))

    assert decoded.sample_rate == 16000
    assert decoded.duration_seconds == pytest.approx(1.0, abs=0.05)
    assert decoded.samples.ndim == 1
    assert decoded.samples.size == pytest.approx(16000, abs=200)


def test_decode_webm_opus_browser_recording():
    from tests.helpers import webm_opus_bytes

    data = webm_opus_bytes(seconds=1.2)
    if data is None:
        pytest.skip("local FFmpeg build has no libopus encoder")

    decoded = decode_audio(data)

    assert decoded.container in {"webm", "unknown"}
    assert decoded.duration_seconds == pytest.approx(1.2, abs=0.15)
    assert decoded.samples.size > 0


def test_encode_decode_roundtrip_preserves_duration_and_level():
    samples = tone(0.5, amplitude=0.5)
    decoded = decode_audio(encode_wav(samples, 16000))

    assert decoded.duration_seconds == pytest.approx(0.5, abs=0.02)
    assert decoded.peak == pytest.approx(0.5, abs=0.05)


def test_silence_is_detected():
    decoded = decode_audio(silent_wav_bytes(1.0))

    assert decoded.rms < SILENCE_RMS_THRESHOLD
    assert decoded.is_silent is True


def test_empty_payload_is_rejected_as_undecodable():
    with pytest.raises(AudioDecodeError):
        decode_audio(b"")


def test_garbage_payload_is_rejected_as_undecodable():
    with pytest.raises(AudioDecodeError) as excinfo:
        decode_audio(b"this is definitely not audio" * 10)

    assert "could not be decoded" in str(excinfo.value.detail)


def test_truncated_wav_is_rejected():
    truncated = wav_bytes(1.0)[:60]

    with pytest.raises(AudioDecodeError):
        decode_audio(truncated)


def test_audio_longer_than_limit_is_rejected():
    with pytest.raises(PayloadTooLargeError) as excinfo:
        decode_audio(wav_bytes(3.0), max_seconds=1.0)

    assert "longer than" in excinfo.value.detail


def test_validate_upload_accepts_wav():
    assert validate_upload("clip.wav", wav_bytes(0.2), SETTINGS) == ".wav"


@pytest.mark.parametrize("filename", ["clip.txt", "clip.pdf", "clip.exe"])
def test_validate_upload_rejects_unsupported_extension(filename):
    with pytest.raises(UnsupportedFormatError) as excinfo:
        validate_upload(filename, b"some bytes", SETTINGS)

    assert ".wav" in excinfo.value.detail


def test_validate_upload_rejects_missing_extension():
    with pytest.raises(UnsupportedFormatError):
        validate_upload("recording", wav_bytes(0.2), SETTINGS)


def test_validate_upload_rejects_empty_file():
    with pytest.raises(BadRequestError):
        validate_upload("clip.wav", b"", SETTINGS)

"""Audio validation, format sniffing and decoding.

Browser recordings arrive as WebM/Opus, uploads may be WAV/MP3/M4A/FLAC. Rather
than shelling out to an ``ffmpeg`` binary (which the assessment forbids as a
mandatory native dependency) we decode with PyAV, whose wheels bundle FFmpeg
and which is already a transitive dependency of faster-whisper.

All audio is converted to 16 kHz mono ``float32`` in the range ``[-1, 1]``,
which is exactly what Whisper expects.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from .config import Settings
from .errors import AudioDecodeError, BadRequestError, PayloadTooLargeError, UnsupportedFormatError

#: Whisper's native sample rate.
TARGET_SAMPLE_RATE = 16_000

#: Below this RMS the clip is considered digital silence; running the model on
#: silence only produces hallucinations, so we short-circuit it.
SILENCE_RMS_THRESHOLD = 1e-4

#: Clips shorter than this cannot contain speech. PyAV happily "decodes" a
#: truncated WAV into a handful of padded samples, so this guard also catches
#: corrupted/truncated uploads instead of sending them to the model.
MIN_DECODABLE_SECONDS = 0.1

#: Map of accepted file extensions to the container family they normally carry.
EXTENSION_FAMILIES: dict[str, str] = {
    ".wav": "wav",
    ".webm": "webm",
    ".ogg": "ogg",
    ".oga": "ogg",
    ".opus": "ogg",
    ".mp3": "mp3",
    ".m4a": "mp4",
    ".mp4": "mp4",
    ".flac": "flac",
}

_DECODE_FAILURE_MESSAGE = (
    "The audio file could not be decoded. It may be corrupted, truncated or "
    "encoded in a way this server cannot read. Please try another file."
)


@dataclass(frozen=True)
class DecodedAudio:
    """Container-agnostic representation of a decoded clip."""

    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    container: str
    channels: int = 1

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))

    @property
    def peak(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.max(np.abs(self.samples)))

    @property
    def is_silent(self) -> bool:
        """True when the clip carries no usable signal at all."""

        return self.rms < SILENCE_RMS_THRESHOLD


def detect_container(data: bytes) -> str | None:
    """Best-effort container sniffing based on magic bytes."""

    head = data[:32]
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "webm"  # Matroska / WebM family
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:3] == b"ID3":
        return "mp3"
    if head[:4] == b"FORM" and head[8:12] in {b"AIFF", b"AIFC"}:
        return "aiff"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return "mp4"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"
    if len(head) >= 12 and head[:4] == b"RIFF":
        return "riff"
    return None


def supported_formats_hint(settings: Settings) -> str:
    return ", ".join(sorted(settings.allowed_extensions))


def validate_upload(filename: str | None, data: bytes, settings: Settings) -> str:
    """Validate an upload and return the normalised extension (e.g. ``.wav``).

    Raises:
        BadRequestError: empty payload or missing filename.
        UnsupportedFormatError: extension outside the allow-list.
    """

    if not data:
        raise BadRequestError(
            "The uploaded file is empty (0 bytes). Please select a non-empty audio file."
        )

    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        raise UnsupportedFormatError(
            "The uploaded file has no file extension, so its format cannot be "
            f"determined. Supported formats: {supported_formats_hint(settings)}."
        )
    if suffix not in settings.allowed_extensions:
        raise UnsupportedFormatError(
            f"Unsupported audio format '{suffix}'. Supported formats: "
            f"{supported_formats_hint(settings)}."
        )
    return suffix


def _resample_chunk(resampler: "av.AudioResampler", frame: "av.AudioFrame | None") -> list:
    try:
        produced = resampler.resample(frame)
    except Exception as exc:  # pragma: no cover - depends on FFmpeg internals
        raise AudioDecodeError(_DECODE_FAILURE_MESSAGE) from exc
    if produced is None:
        return []
    if isinstance(produced, list):
        return produced
    return [produced]


def decode_audio(
    data: bytes,
    *,
    max_seconds: float | None = None,
    max_rate: int = TARGET_SAMPLE_RATE,
) -> DecodedAudio:
    """Decode *data* into 16 kHz mono float32 samples.

    Args:
        data: Raw bytes of the uploaded/streamed audio file.
        max_seconds: Optional hard limit on decoded duration.
        max_rate: Sample rate to resample to (kept configurable for tests).

    Raises:
        AudioDecodeError: when the bytes are not decodable audio.
        PayloadTooLargeError: when the decoded clip is longer than *max_seconds*.
    """

    if not data:
        raise AudioDecodeError("No audio data was received.")

    container_kind = detect_container(data) or "unknown"
    parts: list[np.ndarray] = []
    total = 0

    try:
        container = av.open(io.BytesIO(data), mode="r")
    except Exception as exc:
        raise AudioDecodeError(_DECODE_FAILURE_MESSAGE) from exc

    try:
        try:
            stream = next(s for s in container.streams if s.type == "audio")
        except StopIteration as exc:
            raise AudioDecodeError(
                "The uploaded file does not contain a decodable audio stream."
            ) from exc

        resampler = av.AudioResampler(format="flt", layout="mono", rate=max_rate)
        frames: list = []
        try:
            for frame in container.decode(stream):
                frames.extend(_resample_chunk(resampler, frame))
            frames.extend(_resample_chunk(resampler, None))
        except AudioDecodeError:
            raise
        except Exception as exc:
            raise AudioDecodeError(_DECODE_FAILURE_MESSAGE) from exc

        for frame in frames:
            array = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
            if array.size == 0:
                continue
            parts.append(array)
            total += array.size
            if max_seconds is not None and total / max_rate > max_seconds:
                raise PayloadTooLargeError(
                    f"The audio is longer than the {int(max_seconds)} s limit "
                    f"supported by this server."
                )
    finally:
        container.close()

    if total == 0:
        raise AudioDecodeError(
            "The uploaded file does not contain any decodable audio samples."
        )

    if total / max_rate < MIN_DECODABLE_SECONDS:
        raise AudioDecodeError(
            "The audio is too short to contain speech (under 0.1 s). The file "
            "may be corrupted or truncated."
        )

    samples = np.concatenate(parts).astype(np.float32, copy=False)
    np.clip(samples, -1.0, 1.0, out=samples)

    return DecodedAudio(
        samples=samples,
        sample_rate=max_rate,
        duration_seconds=round(total / max_rate, 3),
        container=container_kind,
        channels=1,
    )


def encode_wav(samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Encode mono float32 samples as a 16-bit PCM WAV byte string.

    Used by the sample generator and by the test-suite to build fixtures.
    """

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()

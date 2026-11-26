"""Audio fixtures and text-comparison helpers shared by the tests."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np

from app.audio import encode_wav

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"


def tone(
    seconds: float = 2.0,
    sample_rate: int = 16_000,
    frequency: float = 220.0,
    amplitude: float = 0.4,
) -> np.ndarray:
    """A deterministic non-silent signal (recognisable by the silence gate)."""

    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    # Add a slow amplitude envelope so the clip is not a pure tone.
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t)
    return (wave * envelope).astype(np.float32)


def wav_bytes(
    seconds: float = 2.0,
    sample_rate: int = 16_000,
    frequency: float = 220.0,
    amplitude: float = 0.4,
) -> bytes:
    return encode_wav(tone(seconds, sample_rate, frequency, amplitude), sample_rate)


def silent_wav_bytes(seconds: float = 2.0, sample_rate: int = 16_000) -> bytes:
    return encode_wav(np.zeros(int(seconds * sample_rate), dtype=np.float32), sample_rate)


def stereo_wav_bytes(seconds: float = 1.0, sample_rate: int = 44_100) -> bytes:
    """44.1 kHz stereo WAV, i.e. a file a user might really upload."""

    samples = tone(seconds, sample_rate)
    pcm = (np.clip(samples, -1, 1) * 32767.0).astype("<i2")
    interleaved = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    buffer = io.BytesIO()
    import wave

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(interleaved.tobytes())
    return buffer.getvalue()


def webm_opus_bytes(seconds: float = 1.5, sample_rate: int = 16_000) -> bytes | None:
    """Encode a clip the way MediaRecorder does (WebM/Opus).

    Returns ``None`` when the local PyAV build has no libopus encoder, so the
    caller can skip instead of failing.
    """

    try:
        import av
    except ImportError:  # pragma: no cover - av is a hard dependency
        return None

    samples = tone(seconds, sample_rate)
    buffer = io.BytesIO()
    try:
        container = av.open(buffer, mode="w", format="webm")
        stream = container.add_stream("libopus", rate=sample_rate, layout="mono")
        stream.bit_rate = 64_000

        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = sample_rate
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
    except Exception:
        return None
    data = buffer.getvalue()
    return data or None


def post_audio(client, audio: bytes, language: str | None = "zh", filename: str = "clip.wav", content_type: str = "audio/wav"):
    """POST an audio clip to /api/transcribe the way the browser does."""

    data = {} if language is None else {"language": language}
    return client.post(
        "/api/transcribe",
        files={"file": (filename, audio, content_type)},
        data=data,
    )


# --------------------------------------------------------------------------- #
# reference comparison
# --------------------------------------------------------------------------- #

_PUNCTUATION = re.compile(r"[\s,，。.!！?？、;；:：'\"“”‘’()（）\-—…·]+")


def normalise_for_cer(text: str) -> str:
    return _PUNCTUATION.sub("", text).lower()


def normalise_for_wer(text: str) -> list[str]:
    return [token for token in _PUNCTUATION.sub(" ", text.lower()).split(" ") if token]


def edit_distance(source: list, target: list) -> int:
    """Levenshtein distance between two sequences of items."""

    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_item in enumerate(source, start=1):
        current = [i]
        for j, target_item in enumerate(target, start=1):
            cost = 0 if source_item == target_item else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(normalise_for_cer(reference))
    hyp = list(normalise_for_cer(hypothesis))
    if not ref:
        return 0.0
    return edit_distance(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalise_for_wer(reference)
    hyp = normalise_for_wer(hypothesis)
    if not ref:
        return 0.0
    return edit_distance(ref, hyp) / len(ref)


def load_reference_texts() -> dict[str, str]:
    """Parse samples/reference.txt into ``{filename: transcript}``."""

    reference_file = SAMPLES_DIR / "reference.txt"
    mapping: dict[str, str] = {}
    for line in reference_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        filename, _, transcript = line.partition("\t")
        mapping[filename.strip()] = transcript.strip()
    return mapping

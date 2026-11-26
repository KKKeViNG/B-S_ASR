"""Prove that recognition works with all outbound network access disabled.

The assessment requires that once the weights are prepared, recognition runs
without internet. This script hard-disables ``socket`` connections in the
process, loads the model from the local cache with ``STT_OFFLINE=1`` semantics
and transcribes the bundled Chinese and English samples.

Run after ``python scripts/download_model.py``::

    python scripts/offline_check.py
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.audio import decode_audio  # noqa: E402
from app.config import Settings  # noqa: E402


def block_network() -> None:
    """Make every attempt to open a network connection fail loudly."""

    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is disabled by scripts/offline_check.py")

    socket.socket.connect = _blocked  # type: ignore[assignment]
    socket.socket.connect_ex = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]


def main() -> int:
    samples = {
        "chinese.wav": "zh",
        "english.wav": "en",
    }

    for filename in samples:
        if not (PROJECT_ROOT / "samples" / filename).is_file():
            print(f"Missing sample: samples/{filename}", file=sys.stderr)
            return 1

    block_network()
    print("Network disabled (socket.connect / getaddrinfo blocked).")

    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    settings = Settings(offline=True)
    print(f"Model: {settings.model_display_name}, cache: {settings.model_dir}")

    from app.asr.whisper_engine import WhisperEngine

    engine = WhisperEngine(settings)
    started = time.perf_counter()
    engine.load()
    print(f"Model loaded offline in {time.perf_counter() - started:.2f}s")

    failures = 0
    for filename, language in samples.items():
        decoded = decode_audio((PROJECT_ROOT / "samples" / filename).read_bytes())
        started = time.perf_counter()
        result = engine.transcribe(decoded.samples, decoded.sample_rate, language)
        elapsed = time.perf_counter() - started
        print(
            f"[{language}] {filename} ({decoded.duration_seconds:.2f}s audio) "
            f"-> {elapsed:.2f}s, {len(result.text)} chars: {result.text}"
        )
        if not result.text.strip():
            failures += 1

    if failures:
        print(f"Offline check failed for {failures} sample(s).", file=sys.stderr)
        return 1

    print("Offline recognition verified for Chinese and English.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

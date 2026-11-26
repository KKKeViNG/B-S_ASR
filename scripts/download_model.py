"""Download the local ASR weights once, then run fully offline.

Usage (from the repository root, after installing requirements.txt)::

    python scripts/download_model.py                # pick the fastest reachable mirror
    python scripts/download_model.py --check        # also transcribe samples/chinese.wav
    python scripts/download_model.py --offline      # verify an existing cache without network

The weights are stored under ``models/`` (git-ignored). Once this command
succeeds, set ``STT_OFFLINE=1`` to guarantee that recognition never touches the
network again::

    $env:STT_OFFLINE = "1"
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402

ENDPOINTS = {
    "huggingface": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}

PROBE_PATH = "/api/models/Systran/faster-whisper-small"


def _reachable(endpoint: str, timeout: float = 6.0) -> bool:
    try:
        with urllib.request.urlopen(endpoint + PROBE_PATH, timeout=timeout) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


def configure_endpoint(mirror: str) -> str:
    """Choose the download endpoint and export ``HF_ENDPOINT`` if needed."""

    if mirror == "huggingface":
        return "huggingface"
    if mirror == "hf-mirror":
        os.environ["HF_ENDPOINT"] = ENDPOINTS["hf-mirror"]
        return "hf-mirror"

    if _reachable(ENDPOINTS["huggingface"]):
        return "huggingface"
    if _reachable(ENDPOINTS["hf-mirror"]):
        os.environ["HF_ENDPOINT"] = ENDPOINTS["hf-mirror"]
        return "hf-mirror"
    raise SystemExit(
        "Neither huggingface.co nor hf-mirror.com is reachable. "
        "Check the network connection, or copy a model directory into "
        f"{Settings().model_dir} manually (see README)."
    )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the local faster-whisper weights")
    parser.add_argument("--model", default=None, help="model size or local path (default: STT_MODEL_SIZE)")
    parser.add_argument("--model-dir", default=None, help="cache directory (default: STT_MODEL_DIR or ./models)")
    parser.add_argument("--device", default=None, help="cpu / cuda (default: STT_DEVICE or cpu)")
    parser.add_argument("--compute-type", default=None, help="int8 / float32 (default: STT_COMPUTE_TYPE)")
    parser.add_argument(
        "--mirror",
        choices=["auto", "huggingface", "hf-mirror"],
        default="auto",
        help="download endpoint (auto probes huggingface.co first, then hf-mirror.com)",
    )
    parser.add_argument("--offline", action="store_true", help="use only the local cache")
    parser.add_argument("--check", action="store_true", help="transcribe samples/chinese.wav afterwards")
    args = parser.parse_args()

    settings = Settings.from_env()
    model = args.model or settings.model_size
    model_dir = Path(args.model_dir or settings.model_dir)
    device = args.device or settings.device
    compute_type = args.compute_type or settings.compute_type
    model_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print(f"Offline mode: using the local cache in {model_dir}")
    else:
        endpoint = configure_endpoint(args.mirror)
        print(f"Download endpoint: {endpoint}" + (f" ({os.environ.get('HF_ENDPOINT')})" if os.environ.get("HF_ENDPOINT") else ""))

    print(f"Preparing model '{model}' (device={device}, compute_type={compute_type}) ...")
    started = time.perf_counter()

    # Imported lazily so the environment variables above take effect first.
    from faster_whisper import WhisperModel

    try:
        engine = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=str(model_dir),
            local_files_only=args.offline,
        )
    except Exception as exc:
        print(f"\nModel preparation failed: {exc}", file=sys.stderr)
        print(
            "If the download was interrupted, re-run this script; "
            "for network restrictions try --mirror hf-mirror.",
            file=sys.stderr,
        )
        return 1

    elapsed = time.perf_counter() - started
    size_mb = _directory_size(model_dir) / (1024 * 1024)
    print(f"Model ready in {elapsed:.1f}s. Cache: {model_dir} ({size_mb:.1f} MB)")

    if args.check:
        sys.path.insert(0, str(PROJECT_ROOT))
        from app.audio import decode_audio

        sample = PROJECT_ROOT / "samples" / "chinese.wav"
        if not sample.is_file():
            print(f"Skipping --check: {sample} not found", file=sys.stderr)
            return 0
        decoded = decode_audio(sample.read_bytes())
        check_started = time.perf_counter()
        segments, info = engine.transcribe(decoded.samples, language="zh", beam_size=5)
        text = "".join(segment.text for segment in segments).strip()
        print(
            f"Check transcription ({decoded.duration_seconds:.2f}s audio, "
            f"{time.perf_counter() - check_started:.2f}s): {text}"
        )
        if not text:
            print("Transcription returned no text - please inspect the model cache.", file=sys.stderr)
            return 1

    print("Done. Start the service with:")
    print("  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

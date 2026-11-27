"""Runtime configuration for the local speech-to-text service.

Every knob can be overridden with an environment variable so that the service
stays portable: the assessment requires a plain ``pip install`` + ``uvicorn``
start-up on a CPU-only Windows machine, without Docker, Node.js or a database.

Example (PowerShell)::

    $env:STT_MODEL_SIZE = "base"
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Repository root (``.../local-speech-to-text``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Default directory for downloaded model weights (git-ignored).
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

#: Languages accepted by the API. ``auto`` is the bonus language-identification
#: mode; ``zh`` and ``en`` are the two required modes.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("zh", "en", "auto")

#: Audio containers accepted by the upload endpoint. ``.wav`` is mandatory,
#: ``.webm``/``.ogg``/``.opus`` cover what a browser actually records, the rest
#: are common files users may have on disk.
SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".wav",
    ".webm",
    ".ogg",
    ".oga",
    ".opus",
    ".mp3",
    ".m4a",
    ".mp4",
    ".flac",
)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env(name: str) -> str | None:
    """Return a stripped environment value, or ``None`` when unset/blank."""

    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _env_str(name: str, default: str) -> str:
    return _env(name) or default


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Environment variable {name} must be a number, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings, built with :meth:`from_env`."""

    # ---- Model / inference ----
    model_size: str = "small"
    model_dir: Path = DEFAULT_MODEL_DIR
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    cpu_threads: int = 0
    offline: bool = False
    vad_filter: bool = False
    preload: bool = True
    #: Word-level timings cost roughly 2-2.5x inference time; off by default.
    word_timestamps: bool = False

    # ---- Request handling ----
    default_language: str = "zh"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_audio_seconds: float = 120.0
    #: 0 means "derive from the audio duration" (4x audio + 15 s, capped).
    request_timeout_seconds: float = 0.0
    #: Optional smaller model used once when inference fails, e.g. "base".
    fallback_model: str = ""

    # ---- Microphone recording / streaming (bonus feature) ----
    max_recording_seconds: float = 60.0
    streaming_enabled: bool = True
    streaming_chunk_seconds: float = 2.5
    streaming_max_seconds: float = 60.0
    max_stream_chunk_bytes: int = 8 * 1024 * 1024

    # ---- Validation ----
    allowed_extensions: frozenset[str] = field(
        default_factory=lambda: frozenset(SUPPORTED_EXTENSIONS)
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``STT_*`` environment variables."""

        return cls(
            model_size=_env_str("STT_MODEL_SIZE", cls.model_size),
            model_dir=Path(_env_str("STT_MODEL_DIR", str(cls.model_dir))),
            device=_env_str("STT_DEVICE", cls.device),
            compute_type=_env_str("STT_COMPUTE_TYPE", cls.compute_type),
            beam_size=_env_int("STT_BEAM_SIZE", cls.beam_size),
            cpu_threads=_env_int("STT_CPU_THREADS", cls.cpu_threads),
            offline=_env_bool("STT_OFFLINE", cls.offline),
            vad_filter=_env_bool("STT_VAD_FILTER", cls.vad_filter),
            preload=_env_bool("STT_PRELOAD", cls.preload),
            word_timestamps=_env_bool("STT_WORD_TIMESTAMPS", cls.word_timestamps),
            default_language=_env_str("STT_DEFAULT_LANGUAGE", cls.default_language),
            max_upload_bytes=_env_int("STT_MAX_UPLOAD_BYTES", cls.max_upload_bytes),
            max_audio_seconds=_env_float("STT_MAX_AUDIO_SECONDS", cls.max_audio_seconds),
            request_timeout_seconds=_env_float(
                "STT_REQUEST_TIMEOUT_SECONDS", cls.request_timeout_seconds
            ),
            fallback_model=_env_str("STT_FALLBACK_MODEL", cls.fallback_model),
            max_recording_seconds=_env_float(
                "STT_MAX_RECORDING_SECONDS", cls.max_recording_seconds
            ),
            streaming_enabled=_env_bool("STT_STREAMING_ENABLED", cls.streaming_enabled),
            streaming_chunk_seconds=_env_float(
                "STT_STREAMING_CHUNK_SECONDS", cls.streaming_chunk_seconds
            ),
            streaming_max_seconds=_env_float(
                "STT_STREAMING_MAX_SECONDS", cls.streaming_max_seconds
            ),
            max_stream_chunk_bytes=_env_int(
                "STT_MAX_STREAM_CHUNK_BYTES", cls.max_stream_chunk_bytes
            ),
        )

    # ---- Derived helpers -------------------------------------------------

    @property
    def model_display_name(self) -> str:
        """Human readable model identifier used in API responses."""

        return f"faster-whisper/{self.model_size}"

    @property
    def max_upload_mb(self) -> float:
        return round(self.max_upload_bytes / (1024 * 1024), 2)

    def timeout_for(self, audio_seconds: float) -> float:
        """Request deadline for one clip (never below 15 s)."""

        if self.request_timeout_seconds > 0:
            return self.request_timeout_seconds
        return max(15.0, min(600.0, audio_seconds * 4.0 + 15.0))

    def public_config(self) -> dict[str, object]:
        """Configuration exposed to the browser via ``GET /api/config``."""

        return {
            "max_recording_seconds": self.max_recording_seconds,
            "max_upload_bytes": self.max_upload_bytes,
            "max_upload_mb": self.max_upload_mb,
            "max_audio_seconds": self.max_audio_seconds,
            "languages": list(SUPPORTED_LANGUAGES),
            "default_language": self.default_language,
            "allowed_extensions": sorted(self.allowed_extensions),
            "model": {
                "name": self.model_display_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "beam_size": self.beam_size,
                "offline": self.offline,
                "preload": self.preload,
                "word_timestamps": self.word_timestamps,
                "fallback_model": self.fallback_model or None,
                "request_timeout_seconds": self.request_timeout_seconds,
            },
            "streaming": {
                "enabled": self.streaming_enabled,
                "chunk_seconds": self.streaming_chunk_seconds,
                "max_seconds": self.streaming_max_seconds,
                "endpoint": "/ws/stream",
            },
        }

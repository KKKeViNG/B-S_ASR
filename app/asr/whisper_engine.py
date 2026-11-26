"""faster-whisper (CTranslate2) backend.

Design notes:

* The model is created **lazily** on the first recognition request. Importing
  the app or running ``pytest`` therefore never triggers a model download.
* Model creation and inference are guarded by locks: a CTranslate2 model is not
  safe to use concurrently from several threads, and the FastAPI thread pool
  may call us in parallel.
* Failures are translated into ``ModelUnavailableError`` (weights/backend) or
  ``TranscriptionFailedError`` (inference) so the API can answer with a clear
  5xx instead of a stack trace.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Sequence

import numpy as np

from ..audio import TARGET_SAMPLE_RATE
from ..config import Settings
from ..errors import ModelUnavailableError, TranscriptionFailedError
from .base import TranscriptionResult, TranscriptionSegment, normalize_text

logger = logging.getLogger(__name__)


class WhisperEngine:
    """Local ASR engine backed by faster-whisper on CPU (GPU optional)."""

    name = "faster-whisper"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._load_seconds: float | None = None
        self._last_error: str | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        """Load (and cache) the CTranslate2 model, downloading weights if needed."""

        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model
            started = time.perf_counter()
            try:
                from faster_whisper import WhisperModel

                model = WhisperModel(
                    self._settings.model_size,
                    device=self._settings.device,
                    compute_type=self._settings.compute_type,
                    download_root=str(self._settings.model_dir),
                    cpu_threads=self._settings.cpu_threads,
                    local_files_only=self._settings.offline,
                )
            except Exception as exc:  # ImportError, missing weights, bad backend...
                self._last_error = str(exc)
                logger.warning("Failed to load ASR model: %s", exc, exc_info=True)
                raise ModelUnavailableError(
                    "The local speech recognition model could not be loaded "
                    f"(model: {self._settings.model_display_name}). "
                    f"Reason: {exc}. "
                    "If the weights have never been downloaded, run "
                    "`python scripts/download_model.py` once while online."
                ) from exc

            self._model = model
            self._load_seconds = round(time.perf_counter() - started, 3)
            self._last_error = None
            logger.info(
                "ASR model %s loaded in %.2fs", self._settings.model_display_name, self._load_seconds
            )
            return model

    # ---- inference -------------------------------------------------------

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        """Recognise a fully-buffered clip; *language* ``None`` means auto-detect."""

        if sample_rate != TARGET_SAMPLE_RATE:
            raise TranscriptionFailedError(
                f"Unexpected sample rate {sample_rate} Hz; the ASR engine requires "
                f"{TARGET_SAMPLE_RATE} Hz mono audio."
            )
        if samples.size == 0:
            raise TranscriptionFailedError("No audio samples were provided for recognition.")

        model = self.load()
        audio = np.ascontiguousarray(samples, dtype=np.float32)

        with self._inference_lock:
            try:
                segment_iter, info = model.transcribe(
                    audio,
                    language=language,
                    task="transcribe",
                    beam_size=self._settings.beam_size,
                    vad_filter=self._settings.vad_filter,
                    condition_on_previous_text=False,
                    without_timestamps=False,
                )
                # The generator streams results lazily, so it must be consumed
                # while the inference lock is held.
                segments: list[TranscriptionSegment] = [
                    TranscriptionSegment(
                        start=round(float(segment.start), 2),
                        end=round(float(segment.end), 2),
                        text=segment.text,
                    )
                    for segment in segment_iter
                ]
            except Exception as exc:
                logger.warning("Transcription failed: %s", exc, exc_info=True)
                raise TranscriptionFailedError(
                    f"Speech recognition failed while processing this audio: {exc}"
                ) from exc

        # faster-whisper prefixes English segments with a space, hence the plain
        # concatenation followed by whitespace normalisation.
        text = normalize_text("".join(segment.text for segment in segments))
        detected = getattr(info, "language", None)
        probability = getattr(info, "language_probability", None)

        return TranscriptionResult(
            text=text,
            language=language or detected or "unknown",
            detected_language=detected,
            language_probability=round(float(probability), 4) if probability is not None else None,
            segments=tuple(segments),
        )

    # ---- introspection ---------------------------------------------------

    def status(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self._settings.model_display_name,
            "device": self._settings.device,
            "compute_type": self._settings.compute_type,
            "beam_size": self._settings.beam_size,
            "offline": self._settings.offline,
            "loaded": self.is_loaded,
            "load_seconds": self._load_seconds,
            "last_error": self._last_error,
        }


def build_engine(settings: Settings) -> WhisperEngine:
    """Create the production ASR engine for *settings*."""

    return WhisperEngine(settings)

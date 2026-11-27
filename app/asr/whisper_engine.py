"""faster-whisper (CTranslate2) backend.

Design notes:

* The model is created **lazily** on the first recognition request. Importing
  the app or running ``pytest`` therefore never triggers a model download.
* Model creation and inference are guarded by locks: a CTranslate2 model is not
  safe to use concurrently from several threads, and the FastAPI thread pool
  may call us in parallel.
* The model can be swapped at runtime (``set_model``) without restarting the
  service; the next request loads the new weights.
* Failures are translated into ``ModelUnavailableError`` (weights/backend),
  ``TranscriptionFailedError`` (inference) or ``RecognitionTimeoutError``
  (deadline) so the API can answer with a clear status instead of a stack trace.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Sequence

import numpy as np

from ..audio import TARGET_SAMPLE_RATE
from ..config import Settings
from ..errors import ModelUnavailableError, RecognitionTimeoutError, TranscriptionFailedError
from ..models import is_model_cached
from .base import TranscriptionResult, TranscriptionSegment, Word, normalize_text

logger = logging.getLogger(__name__)


def _segment_words(segment: Any) -> tuple[Word, ...]:
    """Convert faster-whisper word timings into our immutable ``Word`` model."""

    raw_words = getattr(segment, "words", None) or ()
    words: list[Word] = []
    for word in raw_words:
        text = getattr(word, "word", None)
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
        if text is None or start is None or end is None:
            continue
        probability = getattr(word, "probability", None)
        words.append(
            Word(
                word=str(text),
                start=float(start),
                end=float(end),
                probability=float(probability) if probability is not None else None,
            )
        )
    return tuple(words)


class WhisperEngine:
    """Local ASR engine backed by faster-whisper on CPU (GPU optional)."""

    name = "faster-whisper"

    def __init__(self, settings: Settings, model_id: str | None = None) -> None:
        self._settings = settings
        self._model_id = model_id or settings.model_size
        self._model: Any | None = None
        self._load_seconds: float | None = None
        self._last_error: str | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def set_model(self, model_id: str) -> None:
        """Point the engine at another model; the next request loads it.

        Takes the inference lock so an in-flight transcription finishes before
        the weights reference is dropped.
        """

        if model_id == self._model_id and self._model is not None:
            return
        with self._inference_lock:
            with self._load_lock:
                if model_id != self._model_id:
                    logger.info("Switching ASR model %s -> %s", self._model_id, model_id)
                self._model_id = model_id
                self._model = None
                self._load_seconds = None
                self._last_error = None

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
                    self._model_id,
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
                    f"(model: faster-whisper/{self._model_id}). "
                    f"Reason: {exc}. "
                    "If the weights have never been downloaded, run "
                    "`python scripts/download_model.py` once while online."
                ) from exc

            self._model = model
            self._load_seconds = round(time.perf_counter() - started, 3)
            self._last_error = None
            logger.info("ASR model faster-whisper/%s loaded in %.2fs", self._model_id, self._load_seconds)
            return model

    def availability(self) -> dict[str, object]:
        """Describe whether the current model can be used right now."""

        if self._model is not None:
            return {"available": True, "downloaded": True, "reason": "loaded"}
        if is_model_cached(self._settings.model_dir, self._model_id):
            return {"available": True, "downloaded": True, "reason": "cached"}
        if self._settings.offline:
            return {"available": False, "downloaded": False, "reason": "missing_weights_offline"}
        return {"available": True, "downloaded": False, "reason": "will_download"}

    # ---- inference -------------------------------------------------------

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
        *,
        deadline: float | None = None,
        word_timestamps: bool | None = None,
    ) -> TranscriptionResult:
        """Recognise a fully-buffered clip; *language* ``None`` means auto-detect."""

        if sample_rate != TARGET_SAMPLE_RATE:
            raise TranscriptionFailedError(
                f"Unexpected sample rate {sample_rate} Hz; the ASR engine requires "
                f"{TARGET_SAMPLE_RATE} Hz mono audio."
            )
        if samples.size == 0:
            raise TranscriptionFailedError("No audio samples were provided for recognition.")
        if deadline is not None and time.perf_counter() >= deadline:
            raise RecognitionTimeoutError(
                "The recognition deadline was reached before the model could start."
            )

        effective_word_timestamps = (
            self._settings.word_timestamps if word_timestamps is None else bool(word_timestamps)
        )

        model = self.load()
        audio = np.ascontiguousarray(samples, dtype=np.float32)

        with self._inference_lock:
            truncated = False
            segments: list[TranscriptionSegment] = []
            try:
                segment_iter, info = model.transcribe(
                    audio,
                    language=language,
                    task="transcribe",
                    beam_size=self._settings.beam_size,
                    vad_filter=self._settings.vad_filter,
                    condition_on_previous_text=False,
                    word_timestamps=effective_word_timestamps,
                    without_timestamps=False,
                )
                # The generator streams results lazily, so it must be consumed
                # while the inference lock is held.
                for segment in segment_iter:
                    segments.append(
                        TranscriptionSegment(
                            start=round(float(segment.start), 2),
                            end=round(float(segment.end), 2),
                            text=segment.text,
                            words=_segment_words(segment),
                        )
                    )
                    if deadline is not None and time.perf_counter() > deadline:
                        truncated = True
                        logger.warning(
                            "Recognition deadline reached after %d segment(s); returning partial text",
                            len(segments),
                        )
                        break
            except (TranscriptionFailedError, RecognitionTimeoutError):
                raise
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
            model=self._model_id,
            truncated=truncated,
        )

    # ---- introspection ---------------------------------------------------

    def status(self) -> dict[str, object]:
        availability = self.availability()
        return {
            "name": self.name,
            "model": f"faster-whisper/{self._model_id}",
            "model_id": self._model_id,
            "device": self._settings.device,
            "compute_type": self._settings.compute_type,
            "beam_size": self._settings.beam_size,
            "offline": self._settings.offline,
            "word_timestamps": self._settings.word_timestamps,
            "loaded": self.is_loaded,
            "load_seconds": self._load_seconds,
            "last_error": self._last_error,
            "downloaded": availability["downloaded"],
            "available": availability["available"],
            "reason": availability["reason"],
        }


def build_engine(settings: Settings, model_id: str | None = None) -> WhisperEngine:
    """Create the production ASR engine for *settings*."""

    return WhisperEngine(settings, model_id=model_id)

"""ASR engine abstractions.

The HTTP layer only depends on the :class:`ASREngine` protocol, which keeps the
real faster-whisper backend replaceable by a deterministic fake in unit tests
(the assessment explicitly asks for mock-based tests *plus* a separate real
model verification).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Word:
    """A single timed word inside a segment."""

    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass(frozen=True)
class TranscriptionSegment:
    """One timed segment produced by the model."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass(frozen=True)
class TranscriptionResult:
    """Result of one recognition request."""

    text: str
    language: str
    detected_language: str | None = None
    language_probability: float | None = None
    segments: Sequence[TranscriptionSegment] = ()
    #: Model identifier that actually produced this result (may be a fallback).
    model: str | None = None
    #: True when the request deadline stopped decoding before the audio ended.
    truncated: bool = False


@runtime_checkable
class ASREngine(Protocol):
    """Minimal interface the API needs from a speech recogniser."""

    name: str

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        language: str | None,
        *,
        deadline: float | None = None,
        word_timestamps: bool | None = None,
    ) -> TranscriptionResult:
        """Recognise *samples* (mono float32 in ``[-1, 1]``).

        Args:
            samples: 1-D mono audio.
            sample_rate: Sample rate of *samples* in Hz.
            language: ``"zh"``, ``"en"`` or ``None`` to auto-detect.
            deadline: Optional ``time.perf_counter()`` value. Engines should stop
                decoding once it is reached and flag the result as truncated.
            word_timestamps: Per-request override for word-level timings; ``None``
                keeps the configured default.
        """

    def status(self) -> dict[str, object]:
        """Return a JSON-serialisable description of the backend state."""


def normalize_text(text: str) -> str:
    """Collapse whitespace produced by concatenating model segments."""

    return " ".join(text.split())

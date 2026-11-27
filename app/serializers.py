"""Serialise engine results into the JSON shapes used by the API."""

from __future__ import annotations

from typing import Any, Sequence

from .asr.base import TranscriptionSegment


def segment_payloads(segments: Sequence[TranscriptionSegment]) -> list[dict[str, Any]]:
    """Convert engine segments (with optional word timings) to API payloads."""

    return [
        {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": segment.text,
            "words": [
                {
                    "word": word.word,
                    "start": round(float(word.start), 3),
                    "end": round(float(word.end), 3),
                    "probability": (
                        round(float(word.probability), 4) if word.probability is not None else None
                    ),
                }
                for word in segment.words
            ],
        }
        for segment in segments
    ]

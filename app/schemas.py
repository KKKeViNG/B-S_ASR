"""Pydantic schemas describing the public HTTP contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Response of ``GET /health`` (kept intentionally minimal)."""

    status: str = Field(default="ok", examples=["ok"])


class WordOut(BaseModel):
    """One timed word inside a segment (requires word timestamps)."""

    word: str
    start: float
    end: float
    probability: float | None = None


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    words: list[WordOut] = Field(default_factory=list)


class TranscribeResponse(BaseModel):
    """Response of ``POST /api/transcribe``.

    ``text``/``language``/``duration_seconds``/``processing_seconds`` are the
    required fields; the remaining ones are additive information that does not
    break the required contract.
    """

    model_config = ConfigDict(extra="ignore")

    text: str
    language: str
    duration_seconds: float
    processing_seconds: float
    detected_language: str | None = None
    language_probability: float | None = None
    segments: list[SegmentOut] = Field(default_factory=list)
    model: str | None = None
    warning: str | None = None
    truncated: bool = False


class SubtitleRequest(BaseModel):
    """Request for ``POST /api/subtitles`` (formatting only, no inference)."""

    segments: list[SegmentOut] = Field(min_length=1)
    format: Literal["srt", "vtt", "txt"] = "srt"
    text: str | None = None


class ModelSelectRequest(BaseModel):
    """Request for ``POST /api/models/select``."""

    model: str = Field(min_length=1, examples=["small"])


class ErrorResponse(BaseModel):
    """Uniform error body used for every 4xx/5xx response."""

    code: str = Field(examples=["unsupported_format"])
    detail: str = Field(examples=["Unsupported audio format '.txt'."])
    stage: str = Field(default="request", examples=["decode"])
    retryable: bool = Field(default=False)

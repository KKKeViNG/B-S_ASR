"""Pydantic schemas describing the public HTTP contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Response of ``GET /health`` (kept intentionally minimal)."""

    status: str = Field(default="ok", examples=["ok"])


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str


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


class ErrorResponse(BaseModel):
    """Uniform error body used for every 4xx/5xx response."""

    code: str = Field(examples=["unsupported_format"])
    detail: str = Field(examples=["Unsupported audio format '.txt'."])

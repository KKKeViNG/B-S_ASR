"""Subtitle formatting (SRT / WebVTT) for recognised segments.

Keeping this on the server means one implementation is testable and reusable
by any client (web UI, CLI, or an OpenAI-compatible endpoint).
"""

from __future__ import annotations

from typing import Iterable, Mapping

SUPPORTED_FORMATS: tuple[str, ...] = ("srt", "vtt", "txt")

MEDIA_TYPES = {
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
}


def _clamp(seconds: float) -> float:
    return max(0.0, float(seconds))


def format_timestamp(seconds: float, *, millisecond_separator: str = ",") -> str:
    """Format *seconds* as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""

    total_ms = int(round(_clamp(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millisecond_separator}{millis:03d}"


def _segment_text(segment: Mapping[str, object]) -> str:
    return str(segment.get("text") or "").strip()


def to_srt(segments: Iterable[Mapping[str, object]]) -> str:
    """Render segments as a SubRip document."""

    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        text = _segment_text(segment)
        if not text:
            continue
        start = format_timestamp(float(segment.get("start", 0.0)))
        end = format_timestamp(float(segment.get("end", 0.0)))
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(segments: Iterable[Mapping[str, object]]) -> str:
    """Render segments as a WebVTT document."""

    lines: list[str] = ["WEBVTT", ""]
    for segment in segments:
        text = _segment_text(segment)
        if not text:
            continue
        start = format_timestamp(float(segment.get("start", 0.0)), millisecond_separator=".")
        end = format_timestamp(float(segment.get("end", 0.0)), millisecond_separator=".")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def to_txt(segments: Iterable[Mapping[str, object]], text: str | None = None) -> str:
    """Render plain text (the full transcript, one line per segment)."""

    if text and text.strip():
        return text.strip() + "\n"
    return " ".join(_segment_text(segment) for segment in segments if _segment_text(segment)) + "\n"


def build_subtitles(
    segments: Iterable[Mapping[str, object]],
    fmt: str,
    text: str | None = None,
) -> str:
    """Dispatch to the requested format; unknown formats raise ``ValueError``."""

    segment_list = list(segments)
    normalised = fmt.lower().strip()
    if normalised == "srt":
        return to_srt(segment_list)
    if normalised == "vtt":
        return to_vtt(segment_list)
    if normalised == "txt":
        return to_txt(segment_list, text)
    raise ValueError(f"Unsupported subtitle format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}.")

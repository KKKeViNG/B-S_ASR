"""Near real-time speech-to-text over WebSocket (bonus feature F6).

The protocol is deliberately simple and stateless on the server side:

Client -> server
    {"type": "start", "language": "zh"}   # optional; resets the session text
    <binary>                              # a WAV/WebM slice of ~2.5 s of audio
    {"type": "stop"}                      # optional; ends the session
    {"type": "refine"}                    # optional; next binary frame is the
    <binary>                              # complete clip for a final re-decode

Server -> client
    {"type": "ready",   ...}              # after "start"
    {"type": "partial", "text": "...", "chunk_index": 3, ...}
    {"type": "final",   "text": "...", "refined": false, ...}
    {"type": "final",   "text": "...", "refined": true,  ...}   # after "refine"
    {"type": "error",   "code": "...", "detail": "..."}

Segments are recognised independently and concatenated, which gives a
near-live transcript without requiring a natively streaming model. Slicing cuts
words and punctuation at chunk borders, so the browser can optionally send the
whole clip once more after "stop" ("refine") to produce a clean final
transcript; the live partials stay untouched.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .asr.base import ASREngine, normalize_text
from .audio import decode_audio
from .config import SUPPORTED_LANGUAGES, Settings
from .errors import AppError, InvalidLanguageError
from .serializers import segment_payloads

logger = logging.getLogger(__name__)

#: Chunks shorter than this are ignored (they carry no usable speech).
MIN_CHUNK_SECONDS = 0.2

#: Defence in depth: a single WebSocket chunk must never be an hour of audio.
MAX_CHUNK_SECONDS = 30.0

#: Above this duration the session is finalised automatically.
_AUTO_FINALISE_GRACE_SECONDS = 1.0


class _SessionState:
    """Mutable per-connection state."""

    def __init__(self, settings: Settings, language: str) -> None:
        self.language = language
        self.parts: list[str] = []
        self.chunk_index = 0
        self.processed_seconds = 0.0
        self.processing_seconds = 0.0
        self.detected_language: str | None = None

    def reset(self, language: str) -> None:
        self.language = language
        self.parts.clear()
        self.chunk_index = 0
        self.processed_seconds = 0.0
        self.processing_seconds = 0.0
        self.detected_language = None

    def append(self, text: str) -> None:
        cleaned = normalize_text(text)
        if cleaned:
            self.parts.append(cleaned)

    def composed_text(self) -> str:
        separator = "" if self.language == "zh" else " "
        return separator.join(self.parts).strip()


def _normalise_language(raw: Any, settings: Settings) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return settings.default_language
    language = raw.strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise InvalidLanguageError(
            f"Unsupported language '{raw}'. Supported values: "
            f"{', '.join(SUPPORTED_LANGUAGES)}."
        )
    return language


async def _send(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_json(payload)


def _final_payload(state: _SessionState, settings: Settings, elapsed: float, *, refined: bool = False) -> dict[str, Any]:
    return {
        "type": "final",
        "text": state.composed_text(),
        "language": state.language if state.language != "auto" else (state.detected_language or ""),
        "detected_language": state.detected_language,
        "duration_seconds": round(state.processed_seconds, 3),
        "processing_seconds": round(state.processing_seconds, 3),
        "chunks": state.chunk_index,
        "refined": refined,
        "model": settings.model_display_name,
    }


async def handle_stream(websocket: WebSocket, settings: Settings, engine: ASREngine) -> None:
    """Serve one streaming recognition session until disconnect/stop."""

    if not settings.streaming_enabled:
        await websocket.close(code=1008, reason="Streaming recognition is disabled on this server")
        return

    await websocket.accept()
    state = _SessionState(settings, settings.default_language)
    started = time.perf_counter()
    refine_pending = False

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_frame = message.get("text")
            binary_frame = message.get("bytes")

            if text_frame is not None:
                try:
                    command = json.loads(text_frame)
                except json.JSONDecodeError:
                    await _send(
                        websocket,
                        {"type": "error", "code": "bad_request", "detail": "Control frames must be JSON."},
                    )
                    continue
                if not isinstance(command, dict):
                    await _send(
                        websocket,
                        {"type": "error", "code": "bad_request", "detail": "Control frames must be JSON objects."},
                    )
                    continue

                kind = command.get("type")
                if kind == "start":
                    try:
                        language = _normalise_language(command.get("language"), settings)
                    except AppError as exc:
                        await _send(websocket, {"type": "error", "code": exc.code, "detail": exc.detail})
                        continue
                    state.reset(language)
                    await _send(
                        websocket,
                        {
                            "type": "ready",
                            "language": language,
                            "chunk_seconds": settings.streaming_chunk_seconds,
                            "max_seconds": settings.streaming_max_seconds,
                            "model": settings.model_display_name,
                        },
                    )
                elif kind == "stop":
                    # Send the assembled transcript but keep the socket open: the
                    # browser may follow up with a "refine" frame containing the
                    # complete clip for a cleaner final transcript.
                    await _send(websocket, _final_payload(state, settings, time.perf_counter() - started))
                elif kind == "refine":
                    refine_pending = True
                elif kind == "ping":
                    await _send(websocket, {"type": "pong"})
                else:
                    await _send(
                        websocket,
                        {
                            "type": "error",
                            "code": "bad_request",
                            "detail": f"Unknown control message '{kind}'.",
                        },
                    )
                continue

            if binary_frame is None:
                continue

            if len(binary_frame) > settings.max_stream_chunk_bytes:
                await _send(
                    websocket,
                    {
                        "type": "error",
                        "code": "payload_too_large",
                        "detail": "Audio chunk exceeds the maximum allowed size.",
                    },
                )
                continue

            if refine_pending:
                # The browser re-sends the whole clip so the final transcript is
                # produced from continuous audio instead of concatenated slices.
                refine_pending = False
                refine_started = time.perf_counter()
                try:
                    decoded = await run_in_threadpool(
                        decode_audio,
                        binary_frame,
                        max_seconds=settings.streaming_max_seconds + 1.0,
                    )
                    result = await run_in_threadpool(
                        engine.transcribe,
                        decoded.samples,
                        decoded.sample_rate,
                        None if state.language == "auto" else state.language,
                    )
                except AppError as exc:
                    logger.warning("Refined transcription failed: %s", exc.detail)
                    await _send(
                        websocket,
                        {
                            "type": "error",
                            "code": exc.code,
                            "detail": exc.detail,
                            "fatal": exc.status_code >= 500,
                            "refine": True,
                        },
                    )
                    break

                detected = result.detected_language or state.detected_language
                await _send(
                    websocket,
                    {
                        "type": "final",
                        "text": normalize_text(result.text),
                        "language": state.language if state.language != "auto" else (detected or ""),
                        "detected_language": detected,
                        "duration_seconds": decoded.duration_seconds,
                        "processing_seconds": round(time.perf_counter() - refine_started, 3),
                        "chunks": state.chunk_index,
                        "refined": True,
                        "segments": segment_payloads(result.segments),
                        "model": settings.model_display_name,
                    },
                )
                break

            state.chunk_index += 1
            chunk_started = time.perf_counter()

            try:
                decoded = await run_in_threadpool(
                    decode_audio, binary_frame, max_seconds=MAX_CHUNK_SECONDS
                )
            except AppError as exc:
                await _send(
                    websocket,
                    {
                        "type": "error",
                        "code": exc.code,
                        "detail": exc.detail,
                        "chunk_index": state.chunk_index,
                    },
                )
                continue

            if decoded.duration_seconds < MIN_CHUNK_SECONDS:
                await _send(
                    websocket,
                    {
                        "type": "partial",
                        "text": state.composed_text(),
                        "chunk_index": state.chunk_index,
                        "skipped": "chunk_too_short",
                        "duration_seconds": round(state.processed_seconds, 3),
                        "processing_seconds": 0.0,
                    },
                )
                continue

            state.processed_seconds += decoded.duration_seconds

            if decoded.is_silent:
                await _send(
                    websocket,
                    {
                        "type": "partial",
                        "text": state.composed_text(),
                        "chunk_index": state.chunk_index,
                        "silent": True,
                        "duration_seconds": round(state.processed_seconds, 3),
                        "processing_seconds": round(time.perf_counter() - chunk_started, 3),
                    },
                )
                continue

            try:
                result = await run_in_threadpool(
                    engine.transcribe,
                    decoded.samples,
                    decoded.sample_rate,
                    None if state.language == "auto" else state.language,
                )
            except AppError as exc:
                logger.warning("Streaming chunk %s failed: %s", state.chunk_index, exc.detail)
                await _send(
                    websocket,
                    {
                        "type": "error",
                        "code": exc.code,
                        "detail": exc.detail,
                        "chunk_index": state.chunk_index,
                        "fatal": exc.status_code >= 500,
                    },
                )
                if exc.status_code >= 500:
                    break
                continue

            elapsed_chunk = time.perf_counter() - chunk_started
            state.processing_seconds += elapsed_chunk
            state.append(result.text)
            if result.detected_language:
                state.detected_language = result.detected_language

            await _send(
                websocket,
                {
                    "type": "partial",
                    "text": state.composed_text(),
                    "chunk_index": state.chunk_index,
                    "chunk_text": normalize_text(result.text),
                    "duration_seconds": round(state.processed_seconds, 3),
                    "processing_seconds": round(elapsed_chunk, 3),
                },
            )

            if state.processed_seconds >= settings.streaming_max_seconds + _AUTO_FINALISE_GRACE_SECONDS:
                await _send(websocket, _final_payload(state, settings, time.perf_counter() - started))
                break

    except WebSocketDisconnect:
        logger.debug("Streaming client disconnected")
    except Exception:  # pragma: no cover - defensive
        logger.exception("Unexpected streaming error")
        try:
            await _send(
                websocket,
                {
                    "type": "error",
                    "code": "internal_error",
                    "detail": "Unexpected server error during streaming recognition.",
                    "fatal": True,
                },
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:  # pragma: no cover - already closed
            pass

"""FastAPI application for the local speech-to-text web system.

Endpoints required by the assessment:

================================================  ==========================
GET  /health                                      service health check
POST /api/transcribe                              upload audio and recognise
GET  /                                            web UI
================================================  ==========================

Additional endpoints:

* ``GET  /api/config`` - limits and model metadata for the browser
* ``GET  /api/model``  - model/back-end status
* ``WS   /ws/stream``  - near real-time recognition (bonus feature)

Start with::

    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import __version__
from .asr import ASREngine, build_engine
from .audio import decode_audio, validate_upload
from .config import PROJECT_ROOT, SUPPORTED_LANGUAGES, Settings
from .errors import (
    BadRequestError,
    InvalidLanguageError,
    PayloadTooLargeError,
    register_exception_handlers,
)
from .schemas import ErrorResponse, HealthResponse, SegmentOut, TranscribeResponse
from .streaming import handle_stream

logger = logging.getLogger(__name__)

STATIC_DIR = PROJECT_ROOT / "static"
SAMPLES_DIR = PROJECT_ROOT / "samples"
READ_CHUNK_BYTES = 256 * 1024
_MULTIPART_OVERHEAD_BYTES = 2 * 1024 * 1024
_LANGUAGE_ALIASES = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "zh-hans": "zh",
    "chinese": "zh",
    "cmn": "zh",
    "en": "en",
    "en-us": "en",
    "en_us": "en",
    "english": "en",
    "auto": "auto",
    "detect": "auto",
}


def normalize_language(raw: str | None, settings: Settings) -> str:
    """Map user supplied language codes onto ``zh``/``en``/``auto``."""

    if raw is None or not str(raw).strip():
        return settings.default_language
    key = str(raw).strip().lower()
    language = _LANGUAGE_ALIASES.get(key)
    if language is None:
        raise InvalidLanguageError(
            f"Unsupported language '{raw}'. Supported values: {', '.join(SUPPORTED_LANGUAGES)}."
        )
    return language


def _enforce_content_length(request: Request, settings: Settings) -> None:
    """Reject obviously oversized uploads before reading the body."""

    header = request.headers.get("content-length")
    if not header:
        return
    try:
        declared = int(header)
    except ValueError:
        return
    if declared > settings.max_upload_bytes + _MULTIPART_OVERHEAD_BYTES:
        raise PayloadTooLargeError(
            f"The upload exceeds the maximum size of {settings.max_upload_mb:g} MB."
        )


async def _read_upload(file: UploadFile, limit: int) -> bytes:
    """Read an upload while enforcing *limit* bytes."""

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise PayloadTooLargeError(
                f"The uploaded file exceeds the maximum size of "
                f"{round(limit / (1024 * 1024), 2):g} MB."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _model_cache_present(model_dir: Path) -> bool:
    """True when a CTranslate2 model file is already cached locally."""

    try:
        return any(model_dir.glob("**/model.bin"))
    except OSError:  # pragma: no cover - unreadable path
        return False


def create_app(
    settings: Settings | None = None,
    engine: ASREngine | None = None,
) -> FastAPI:
    """Application factory.

    Args:
        settings: Runtime settings; defaults to :meth:`Settings.from_env`.
        engine: ASR backend; defaults to the faster-whisper engine. Tests inject
            a deterministic fake here so that no model download is triggered.
    """

    settings = settings or Settings.from_env()
    engine = engine or build_engine(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Local STT service %s starting (model=%s, device=%s, compute=%s, offline=%s)",
            __version__,
            settings.model_display_name,
            settings.device,
            settings.compute_type,
            settings.offline,
        )
        if not Path(settings.model_dir).exists():
            logger.info(
                "Model directory %s does not exist yet; weights will be downloaded "
                "on the first recognition request (or run scripts/download_model.py).",
                settings.model_dir,
            )
        else:
            loader = getattr(engine, "load", None)
            if settings.preload and loader is not None and _model_cache_present(Path(settings.model_dir)):
                # Warm the model in the background: start-up stays fast, the
                # first user request does not have to wait for model loading.
                def _warm_up() -> None:
                    try:
                        loader()
                        logger.info("ASR model pre-loaded; the first request will be fast")
                    except Exception as exc:  # pragma: no cover - logged only
                        logger.warning("ASR model pre-load failed: %s", exc)

                threading.Thread(target=_warm_up, name="asr-preload", daemon=True).start()
        yield
        logger.info("Local STT service stopping")

    app = FastAPI(
        title="Local Speech-to-Text",
        description=(
            "Browser/Server speech recognition powered by a locally running "
            "open-source model (faster-whisper, CPU friendly)."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine

    register_exception_handlers(app)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if SAMPLES_DIR.is_dir():
        # Convenience for reviewers: the bundled sample clips are also usable
        # straight from the web UI without a microphone.
        app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")

    @app.middleware("http")
    async def _no_cache_static(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static") or path in {"/health", "/api/config"}:
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    # ---- pages -----------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise BadRequestError(
                "The web UI is missing (static/index.html). Please re-install the project files.",
                code="ui_missing",
            )
        return FileResponse(index_path, media_type="text/html")

    # ---- required API ----------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Liveness probe; returns 200 and ``{"status": "ok"}``."""

        return HealthResponse()

    @app.post(
        "/api/transcribe",
        response_model=TranscribeResponse,
        tags=["recognition"],
        responses={
            400: {"model": ErrorResponse, "description": "Empty or malformed request"},
            413: {"model": ErrorResponse, "description": "File too large"},
            415: {"model": ErrorResponse, "description": "Unsupported audio format"},
            422: {"model": ErrorResponse, "description": "Invalid language or undecodable audio"},
            500: {"model": ErrorResponse, "description": "Inference failure"},
            503: {"model": ErrorResponse, "description": "ASR model unavailable"},
        },
    )
    async def transcribe(
        request: Request,
        file: UploadFile = File(..., description="Audio file: wav, webm, ogg, mp3, m4a, flac"),
        language: str = Form(
            default="",
            description="Recognition language: 'zh', 'en' or 'auto' (empty -> server default)",
        ),
    ) -> TranscribeResponse:
        """Recognise a complete audio clip (after-recording recognition)."""

        state_settings: Settings = request.app.state.settings
        app_engine: ASREngine = request.app.state.engine
        started = time.perf_counter()

        selected_language = normalize_language(language, state_settings)
        _enforce_content_length(request, state_settings)

        filename = file.filename
        data = await _read_upload(file, state_settings.max_upload_bytes)
        await file.close()

        validate_upload(filename, data, state_settings)
        decoded = await run_in_threadpool(
            decode_audio, data, max_seconds=state_settings.max_audio_seconds
        )

        fallback_language = (
            selected_language
            if selected_language in {"zh", "en"}
            else state_settings.default_language
        )

        if decoded.is_silent:
            logger.info(
                "Silent clip (%.2fs) - skipping inference", decoded.duration_seconds
            )
            return TranscribeResponse(
                text="",
                language=fallback_language,
                duration_seconds=decoded.duration_seconds,
                processing_seconds=round(time.perf_counter() - started, 3),
                model=state_settings.model_display_name,
                warning="no_speech_detected",
            )

        result = await run_in_threadpool(
            app_engine.transcribe,
            decoded.samples,
            decoded.sample_rate,
            None if selected_language == "auto" else selected_language,
        )

        response_language = (
            selected_language
            if selected_language in {"zh", "en"}
            else (result.detected_language or state_settings.default_language)
        )
        processing_seconds = round(time.perf_counter() - started, 3)

        logger.info(
            "Transcribed %s clip (%.2fs) in %.2fs -> %d chars",
            response_language,
            decoded.duration_seconds,
            processing_seconds,
            len(result.text),
        )

        return TranscribeResponse(
            text=result.text,
            language=response_language,
            duration_seconds=decoded.duration_seconds,
            processing_seconds=processing_seconds,
            detected_language=result.detected_language,
            language_probability=result.language_probability,
            segments=[
                SegmentOut(start=segment.start, end=segment.end, text=segment.text)
                for segment in result.segments
            ],
            model=state_settings.model_display_name,
            warning=None if result.text else "no_speech_detected",
        )

    # ---- auxiliary API ---------------------------------------------------

    @app.get("/api/config", tags=["system"])
    async def config() -> dict[str, object]:
        """Limits and model information consumed by the web UI."""

        return {**settings.public_config(), "version": __version__}

    @app.get("/api/model", tags=["system"])
    async def model_status() -> dict[str, object]:
        """Report whether the model is loaded and how it is configured."""

        status = dict(engine.status())
        status["supported_languages"] = list(SUPPORTED_LANGUAGES)
        return status

    # ---- bonus: streaming ------------------------------------------------

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket) -> None:
        """Near real-time recognition of microphone audio chunks."""

        await handle_stream(websocket, websocket.app.state.settings, websocket.app.state.engine)

    return app


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)

app = create_app()

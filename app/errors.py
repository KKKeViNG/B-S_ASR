"""Application error types and JSON error responses.

Every failure the API can produce is expressed as an :class:`AppError` with a
stable machine readable ``code``, a human readable ``detail``, a coarse
``stage`` (where the failure happened) and a ``retryable`` hint. The front-end
renders ``detail`` directly and uses ``retryable`` to suggest a retry, so
messages must stay actionable.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for errors that map onto an HTTP response."""

    status_code: int = 500
    code: str = "internal_error"
    #: Coarse failure location: request | decode | model | inference | server.
    stage: str = "request"
    #: Whether the client can reasonably retry the same request.
    retryable: bool = False

    def __init__(
        self,
        detail: str,
        *,
        code: str | None = None,
        stage: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        if code is not None:
            self.code = code
        if stage is not None:
            self.stage = stage
        if retryable is not None:
            self.retryable = retryable

    def to_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "detail": self.detail,
            "stage": self.stage,
            "retryable": self.retryable,
        }


class BadRequestError(AppError):
    """The request is syntactically valid but logically unusable."""

    status_code = 400
    code = "bad_request"
    stage = "request"


class InvalidLanguageError(AppError):
    """The requested recognition language is not supported."""

    status_code = 422
    code = "invalid_language"
    stage = "request"


class InvalidModelError(AppError):
    """The requested model identifier is not in the catalogue."""

    status_code = 422
    code = "invalid_model"
    stage = "model"


class UnsupportedFormatError(AppError):
    """The uploaded file has an extension/container we do not accept."""

    status_code = 415
    code = "unsupported_format"
    stage = "request"


class PayloadTooLargeError(AppError):
    """The upload (or the decoded audio) exceeds the configured limits."""

    status_code = 413
    code = "payload_too_large"
    stage = "request"


class AudioDecodeError(AppError):
    """The file is empty, truncated or otherwise not decodable."""

    status_code = 422
    code = "audio_decode_error"
    stage = "decode"


class ModelUnavailableError(AppError):
    """The ASR model could not be loaded (missing weights, bad backend...)."""

    status_code = 503
    code = "model_unavailable"
    stage = "model"
    retryable = True


class TranscriptionFailedError(AppError):
    """The model was loaded but inference failed for this audio."""

    status_code = 500
    code = "transcription_failed"
    stage = "inference"
    retryable = True


class RecognitionTimeoutError(AppError):
    """Recognition did not finish within the request deadline."""

    status_code = 504
    code = "timeout"
    stage = "inference"
    retryable = True


def _json_error(status_code: int, code: str, detail: str, stage: str, retryable: bool) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": code, "detail": detail, "stage": stage, "retryable": retryable},
    )


def _validation_detail(exc: RequestValidationError) -> str:
    """Flatten FastAPI validation errors into one readable sentence."""

    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        message = error.get("msg", "invalid value")
        parts.append(f"{location}: {message}" if location else message)
    return "Invalid request: " + "; ".join(parts) if parts else "Invalid request."


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the JSON error handlers to *app*."""

    @app.exception_handler(AppError)
    async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:  # noqa: D401
        if exc.status_code >= 500:
            logger.warning("Application error (%s/%s): %s", exc.code, exc.stage, exc.detail)
        return _json_error(exc.status_code, exc.code, exc.detail, exc.stage, exc.retryable)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _json_error(422, "validation_error", _validation_detail(exc), "request", False)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        return _json_error(exc.status_code, "http_error", detail, "request", False)

    @app.exception_handler(Exception)
    async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        logger.exception("Unhandled server error", exc_info=exc)
        return _json_error(
            500,
            "internal_error",
            "Unexpected server error while processing the request.",
            "server",
            True,
        )

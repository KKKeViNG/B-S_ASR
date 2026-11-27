"""Model catalogue, local cache inspection and hot switching.

The assessment only requires one local model, but a single hard-coded tier makes
it impossible to trade accuracy for speed without editing environment variables
and restarting. This module adds a small, dependency-free model manager:

* a curated catalogue (size, parameters, languages, licence),
* inspection of the local HuggingFace cache to know what is already downloaded,
* a persisted selection (``<model_dir>/state.json``) applied to the engine at
  runtime, so switching models does not require a restart,
* a Server-Sent-Events progress stream for "download and activate".
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import Settings
from .errors import InvalidModelError, ModelUnavailableError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Static description of one selectable model tier."""

    id: str
    repo_id: str
    display_name: str
    params: str
    size_mb: int
    languages: tuple[str, ...]
    license: str
    description: str
    recommended: bool = False


#: Curated catalogue. Sizes are approximate download sizes (CTranslate2 int8).
MODEL_CATALOG: dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        id="tiny",
        repo_id="Systran/faster-whisper-tiny",
        display_name="tiny",
        params="39M",
        size_mb=75,
        languages=("zh", "en"),
        license="MIT (OpenAI Whisper weights)",
        description="最快、体积最小，准确率较低，仅适合连通性验证。",
    ),
    "base": ModelInfo(
        id="base",
        repo_id="Systran/faster-whisper-base",
        display_name="base",
        params="74M",
        size_mb=145,
        languages=("zh", "en"),
        license="MIT (OpenAI Whisper weights)",
        description="比 tiny 稳一些，中文仍偏弱；可作为 small 的降级备选。",
    ),
    "small": ModelInfo(
        id="small",
        repo_id="Systran/faster-whisper-small",
        display_name="small",
        params="244M",
        size_mb=484,
        languages=("zh", "en"),
        license="MIT (OpenAI Whisper weights)",
        description="默认档位：CPU 上约 2.5x 实时，中文可用，体积适中。",
        recommended=True,
    ),
    "medium": ModelInfo(
        id="medium",
        repo_id="Systran/faster-whisper-medium",
        display_name="medium",
        params="769M",
        size_mb=1530,
        languages=("zh", "en"),
        license="MIT (OpenAI Whisper weights)",
        description="准确率明显高于 small，CPU 上约慢 2-3 倍，适合批量识别。",
    ),
    "large-v3-turbo": ModelInfo(
        id="large-v3-turbo",
        repo_id="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        display_name="large-v3-turbo",
        params="809M",
        size_mb=1620,
        languages=("zh", "en"),
        license="MIT (OpenAI Whisper weights)",
        description="准确率接近 large-v3、速度远快于 medium，中文/英文效果最好。",
    ),
    "large-v3": ModelInfo(
        id="large-v3",
        repo_id="Systran/faster-whisper-large-v3",
        display_name="large-v3",
        params="1550M",
        size_mb=3090,
        languages=("zh", "en"),
        license="MIT (OpenAI Whisper weights)",
        description="最高准确率，CPU 上很慢，仅建议有 GPU 或超长音频时使用。",
    ),
}

DEFAULT_MODEL_ID = "small"


def catalog_entry(model_id: str) -> ModelInfo:
    """Return the catalogue entry or raise a 422 ``invalid_model`` error."""

    info = MODEL_CATALOG.get(model_id)
    if info is None:
        raise InvalidModelError(
            f"Unknown model '{model_id}'. Available models: "
            f"{', '.join(MODEL_CATALOG)}."
        )
    return info


def model_cache_dir(model_dir: Path | str, model_id: str) -> Path:
    """HuggingFace cache directory for one model (``models--<org>--<repo>``)."""

    info = MODEL_CATALOG.get(model_id)
    repo_id = info.repo_id if info is not None else model_id
    return Path(model_dir) / f"models--{repo_id.replace('/', '--')}"


def directory_size(path: Path) -> int:
    """Total size in bytes of every regular file below *path*."""

    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:  # pragma: no cover - unreadable cache
        return 0


def is_model_cached(model_dir: Path | str, model_id: str) -> bool:
    """True when a CTranslate2 ``model.bin`` for *model_id* exists on disk."""

    cache = model_cache_dir(model_dir, model_id)
    try:
        return any(cache.glob("**/model.bin"))
    except OSError:  # pragma: no cover
        return False


class ModelManager:
    """Owns the selected model, its persisted state and cache inspection."""

    def __init__(self, settings: Settings, engine: Any | None = None) -> None:
        self._settings = settings
        self._engine = engine
        self._state_path = Path(settings.model_dir) / "state.json"
        selected = self._read_state() or settings.model_size
        if selected not in MODEL_CATALOG:
            logger.warning("Ignoring unknown persisted model %r", selected)
            selected = settings.model_size
        if selected not in MODEL_CATALOG:
            selected = DEFAULT_MODEL_ID
        self._selected = selected
        if self._engine is not None and hasattr(self._engine, "set_model"):
            self._engine.set_model(selected)

    # ---- state ---------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def selected(self) -> str:
        return self._selected

    def current_model(self) -> str:
        """Model the engine is configured with (falls back to the selection)."""

        model_id = getattr(self._engine, "model_id", None)
        return str(model_id) if model_id else self._selected

    def display_name(self, model_id: str | None = None) -> str:
        return f"faster-whisper/{model_id or self._selected}"

    def _read_state(self) -> str | None:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        model_id = payload.get("selected_model") if isinstance(payload, dict) else None
        return model_id if isinstance(model_id, str) else None

    def _persist(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps({"selected_model": self._selected}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - read-only model dir
            logger.warning("Could not persist model selection: %s", exc)

    # ---- catalogue -----------------------------------------------------

    def describe(self, model_id: str) -> dict[str, Any]:
        info = catalog_entry(model_id)
        downloaded = is_model_cached(self._settings.model_dir, model_id)
        size_on_disk = directory_size(model_cache_dir(self._settings.model_dir, model_id))
        return {
            "id": info.id,
            "display_name": info.display_name,
            "repo_id": info.repo_id,
            "params": info.params,
            "size_mb": info.size_mb,
            "size_on_disk_mb": round(size_on_disk / (1024 * 1024), 1),
            "languages": list(info.languages),
            "license": info.license,
            "description": info.description,
            "recommended": info.recommended,
            "downloaded": downloaded or size_on_disk > 0,
            "selected": info.id == self._selected,
            "current": info.id == self.current_model(),
        }

    def list_models(self) -> dict[str, Any]:
        return {
            "models": [self.describe(model_id) for model_id in MODEL_CATALOG],
            "selected": self._selected,
            "current": self.current_model(),
            "model_dir": str(self._settings.model_dir),
            "device": self._settings.device,
            "compute_type": self._settings.compute_type,
            "offline": self._settings.offline,
        }

    def select(self, model_id: str) -> dict[str, Any]:
        """Persist *model_id* and point the engine at it (no eager load).

        The next recognition request loads the new weights; the UI calls
        :meth:`download_events` to warm it up with a progress bar.
        """

        catalog_entry(model_id)
        self._selected = model_id
        self._persist()
        if self._engine is not None and hasattr(self._engine, "set_model"):
            self._engine.set_model(model_id)
        logger.info("Model selection changed to %s", model_id)
        return self.describe(model_id)

    # ---- download / warm-up stream -------------------------------------

    def download_events(self, model_id: str) -> Iterator[dict[str, Any]]:
        """Yield progress events while the weights download and load."""

        info = catalog_entry(model_id)
        if self._engine is None or not hasattr(self._engine, "load"):
            raise ModelUnavailableError("This ASR backend cannot download models at runtime.")

        if is_model_cached(self._settings.model_dir, model_id):
            yield {
                "stage": "loading",
                "model": model_id,
                "progress": 1.0,
                "bytes": 0,
                "total_bytes": info.size_mb * 1024 * 1024,
            }

        results: "queue.Queue[dict[str, Any]]" = queue.Queue()

        def worker() -> None:
            started = time.perf_counter()
            try:
                if hasattr(self._engine, "set_model"):
                    self._engine.set_model(model_id)
                self._engine.load()
                results.put(
                    {
                        "stage": "ready",
                        "model": model_id,
                        "progress": 1.0,
                        "load_seconds": round(time.perf_counter() - started, 2),
                    }
                )
            except Exception as exc:  # AppError or an unexpected loader failure
                logger.warning("Model download/load failed for %s: %s", model_id, exc)
                results.put(
                    {
                        "stage": "error",
                        "model": model_id,
                        "code": getattr(exc, "code", "model_unavailable"),
                        "detail": getattr(exc, "detail", str(exc)),
                        "retryable": bool(getattr(exc, "retryable", True)),
                    }
                )

        thread = threading.Thread(target=worker, name=f"model-download-{model_id}", daemon=True)
        thread.start()

        total_bytes = info.size_mb * 1024 * 1024
        cache = model_cache_dir(self._settings.model_dir, model_id)
        yield {
            "stage": "starting",
            "model": model_id,
            "progress": 0.0,
            "bytes": 0,
            "total_bytes": total_bytes,
        }

        while True:
            try:
                event = results.get(timeout=0.5)
            except queue.Empty:
                size = directory_size(cache)
                progress = min(0.99, size / total_bytes) if total_bytes else 0.0
                event = {
                    "stage": "downloading" if progress < 0.99 else "loading",
                    "model": model_id,
                    "progress": round(progress, 3),
                    "bytes": size,
                    "total_bytes": total_bytes,
                }
            yield event
            if event.get("stage") in {"ready", "error"}:
                break

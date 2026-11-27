# AGENTS.md — working agreement for AI coding agents

This repository is a self-contained interview deliverable: a **local, CPU-only
speech-to-text web application** (FastAPI + faster-whisper + vanilla JS). It is
also developed with heavy AI assistance, so this file records the rules an
agent (OpenCode, Claude Code, Codex, …) must follow when changing it.

## Project map

```
app/            FastAPI service (routes, config, audio, ASR engines, subtitles)
app/asr/        engine protocol + faster-whisper backend (lazy load, model switch)
static/         browser UI (native ES modules, no build step)
tests/          pytest suite (mock engine by default, real model opt-in)
scripts/        model download, offline check, sample generation, browser E2E
samples/        bundled zh/en reference clips + transcripts
```

## Commands

```bash
python -m pip install -r requirements.txt
python scripts/download_model.py --check          # first run only (~480 MB)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
python -m pytest -q                               # must stay fast and offline
python -m pytest tests/test_real_model.py --run-real-model -q
python scripts/offline_check.py
python -m pip install -r requirements-dev.txt && python -m playwright install chromium
python scripts/e2e_browser_check.py
```

## Non-negotiables

1. **No new mandatory runtime dependencies** beyond pip wheels: no Docker, no
   Node.js/Bun build step, no database, no CUDA-only requirement, no external
   `ffmpeg` binary. Optional dev tooling (Playwright) stays in
   `requirements-dev.txt`.
2. **Recognition must stay fully local and offline-capable.** Never add a cloud
   ASR/LLM call and never rely on the browser Web Speech API.
3. **Default `pytest` run must not download or load model weights** and must not
   need a microphone. Keep the `real_model` marker for anything that does.
4. **Keep the assessment API contract**: `GET /health`, `POST /api/transcribe`
   (fields `text`, `language`, `duration_seconds`, `processing_seconds`),
   `GET /`. Extra endpoints are additive only.
5. **Do not commit** `models/`, `docs/`, `task.md`, or any local artefact
   (`/docs/` and `/task.md` are git-ignored on purpose).
6. **Keep the service startable with one command** on Windows 11 + Python 3.12:
   `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`.

## Code conventions

- Python: 3.12, type hints on public functions, `from __future__ import annotations`,
  docstrings explain *why*; line length ~100.
- Errors: raise `AppError` subclasses from `app/errors.py` (they carry
  `status_code`, `code`, `stage`, `retryable`); never leak stack traces to HTTP.
- API payloads: Pydantic models in `app/schemas.py`; serialise engine results via
  `app/serializers.py` so REST and WebSocket stay consistent.
- Audio: always 16 kHz mono float32 in the engine; decode through `app/audio.py`
  (PyAV) so browser WebM/Opus and uploads share one path.
- Frontend: ES modules under `static/js/`, no framework, Chinese UI copy in
  `messages.js`, all states derived from `refreshControls()`.
- Tests: name by failure mode (`test_deadline_exceeded_returns_504`), assert both
  status code and machine-readable `code`, keep runtime under a few seconds.

## Change workflow

1. Read `README.md` and the relevant doc string before editing.
2. Write a failing test that captures the behaviour (mock engine where possible).
3. Implement the smallest change that makes it pass.
4. Run `python -m pytest -q`; if the change touches the engine or audio path, run
   the `--run-real-model` selection once.
5. For UI changes, run `scripts/e2e_browser_check.py` (fake microphone) and check
   `docs/e2e-results.json`.
6. Update `README.md` (and the local-only `docs/*.md`) for user-visible changes.
7. Commit with a Conventional-Commits subject in English, e.g.
   `feat(models): add runtime model switching`.

## Things that are intentionally out of scope

Voice cloning, TTS, dubbing, speaker diarisation, user accounts, databases,
mobile layout, and multi-user concurrency. Do not add them to satisfy a review
comment unless the task explicitly asks for it.

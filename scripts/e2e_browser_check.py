"""Optional browser end-to-end check with Playwright (development evidence).

It starts the real service, opens the page in Chromium with a fake microphone
fed by ``samples/chinese.wav``, and exercises the actual user flows:

1. microphone recording (Chinese) → recognition result
2. upload of ``samples/english.wav`` (English) → recognition result
3. real-time streaming mode (Chinese)
4. microphone permission denied → friendly error, page stays alive

Requirements (not needed to run the app or the pytest suite)::

    python -m pip install -r requirements-dev.txt
    python -m playwright install chromium

Usage::

    python scripts/e2e_browser_check.py            # headless, all four scenarios
    python scripts/e2e_browser_check.py --headed   # watch the browser
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = PROJECT_ROOT / "samples"
SCREENSHOTS = PROJECT_ROOT / "docs" / "screenshots"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(base_url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.4)
    raise RuntimeError(f"server did not become healthy at {base_url}/health")


def start_server(port: int) -> tuple[subprocess.Popen, Path]:
    log_file = Path(tempfile.gettempdir()) / f"stt-e2e-{port}.log"
    handle = log_file.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_file


def read_result(page) -> str:
    return page.eval_on_selector("#result-text", "element => element.value")


def wait_for_result(page, error_message: str, timeout: float = 180_000) -> str:
    page.wait_for_function(
        "() => { const el = document.getElementById('result-text'); return el && el.value.trim().length > 0; }",
        timeout=timeout,
    )
    text = read_result(page)
    if not text.strip():
        raise AssertionError(error_message)
    return text


def assert_result_assets(page) -> None:
    """Every successful recognition should expose a player, segments and exports."""

    page.wait_for_selector("#audio-player", state="visible", timeout=15_000)
    page.wait_for_selector("#segment-list .segment", timeout=15_000)
    for button in ("#export-txt", "#export-srt", "#export-vtt"):
        if not page.is_enabled(button):
            raise AssertionError(f"{button} should be enabled after recognition")


def scenario_model_ui(page, base_url: str, results: dict) -> None:
    print("[1/5] model catalogue and switching ...")
    page.goto(base_url + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#service-badge", state="visible")
    page.wait_for_function(
        "() => document.getElementById('service-badge').textContent.includes('服务正常')",
        timeout=30_000,
    )
    page.wait_for_function(
        "() => document.querySelectorAll('#model-select option').length >= 5",
        timeout=30_000,
    )

    options = page.eval_on_selector_all("#model-select option", "els => els.map(el => el.value)")
    selected = page.input_value("#model-select")
    page.click("#model-apply")
    page.wait_for_function(
        "() => { const el = document.getElementById('model-status'); return el && (el.textContent.includes('已切换') || el.textContent.includes('已就绪')); }",
        timeout=60_000,
    )
    page.screenshot(path=str(SCREENSHOTS / "ui-model-select.png"), full_page=True)

    results["model_ui"] = {
        "options": options,
        "selected": selected,
        "status": page.text_content("#model-status"),
    }
    print(f"      options: {options}")
    print(f"      status:  {page.text_content('#model-status')}")


def scenario_record_chinese(page, base_url: str, results: dict) -> None:
    print("[2/5] microphone recording (Chinese) ...")
    page.goto(base_url + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#service-badge", state="visible")
    page.wait_for_function(
        "() => document.getElementById('service-badge').textContent.includes('服务正常')",
        timeout=30_000,
    )

    page.click('[data-language="zh"]')
    page.click("#record-button")
    page.wait_for_function(
        "() => document.getElementById('record-status').textContent.includes('正在录音')",
        timeout=20_000,
    )
    page.wait_for_timeout(6000)
    timer_text = page.text_content("#record-timer")
    page.click("#stop-button")

    text = wait_for_result(page, "recording produced no transcription")
    try:
        assert_result_assets(page)
        player_visible = True
    except AssertionError:
        player_visible = False
        page.screenshot(path=str(SCREENSHOTS / "ui-record-missing-assets.png"), full_page=True)
        raise
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOTS / "ui-record-chinese.png"), full_page=True)

    results["record_zh"] = {
        "text": text,
        "timer": timer_text,
        "duration": page.text_content("#meta-duration"),
        "processing": page.text_content("#meta-processing"),
        "language": page.text_content("#meta-language"),
        "player_visible": player_visible,
        "segments": page.eval_on_selector_all("#segment-list .segment", "els => els.length"),
        "has_chinese": any("\u4e00" <= char <= "\u9fff" for char in text),
    }
    print(f"      text: {text}")
    print(f"      segments: {results['record_zh']['segments']}")


def scenario_upload_english(page, base_url: str, results: dict) -> None:
    print("[3/5] uploading samples/english.wav (English) ...")
    page.goto(base_url + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#service-badge", state="visible")
    page.wait_for_function(
        "() => document.getElementById('service-badge').textContent.includes('服务正常')",
        timeout=30_000,
    )

    page.click('[data-language="en"]')
    page.set_input_files("#file-input", str(SAMPLES / "english.wav"))
    page.wait_for_selector("#file-row", state="visible")
    page.click("#upload-button")

    text = wait_for_result(page, "upload produced no transcription")
    assert_result_assets(page)
    page.screenshot(path=str(SCREENSHOTS / "ui-upload-english.png"), full_page=True)

    with page.expect_download() as download_info:
        page.click("#export-srt")
    download = download_info.value
    subtitle_path = Path(tempfile.gettempdir()) / "e2e-transcript.srt"
    download.save_as(str(subtitle_path))
    subtitle_text = subtitle_path.read_text(encoding="utf-8")
    if not subtitle_text.startswith("1\n") or "-->" not in subtitle_text:
        raise AssertionError("exported SRT does not look like valid SubRip")

    results["upload_en"] = {
        "text": text,
        "duration": page.text_content("#meta-duration"),
        "processing": page.text_content("#meta-processing"),
        "language": page.text_content("#meta-language"),
        "srt_preview": subtitle_text[:120],
    }
    print(f"      text: {text}")
    print(f"      srt:  {subtitle_text.splitlines()[1]}")


def scenario_streaming(page, base_url: str, results: dict) -> None:
    print("[4/5] real-time streaming (Chinese) ...")
    page.goto(base_url + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#service-badge", state="visible")
    page.wait_for_function(
        "() => document.getElementById('service-badge').textContent.includes('服务正常')",
        timeout=30_000,
    )

    page.click('[data-language="zh"]')
    page.click('[data-mode="stream"]')
    page.click("#record-button")

    started = time.time()
    try:
        page.wait_for_function(
            "() => { const el = document.getElementById('result-text'); return el && el.value.trim().length > 0; }",
            timeout=60_000,
        )
    except Exception as error:
        page.screenshot(path=str(SCREENSHOTS / "ui-streaming-failure.png"), full_page=True)
        raise AssertionError(f"streaming produced no partial transcript: {error}") from error

    partial_text = read_result(page)
    first_partial_seconds = round(time.time() - started, 1)
    page.wait_for_timeout(3000)
    page.click("#stop-button")

    page.wait_for_function(
        "() => document.getElementById('result-hint').textContent.includes('实时识别完成')",
        timeout=120_000,
    )
    final_text = read_result(page)
    page.screenshot(path=str(SCREENSHOTS / "ui-streaming.png"), full_page=True)

    results["streaming_zh"] = {
        "partial_text": partial_text,
        "final_text": final_text,
        "first_partial_seconds": first_partial_seconds,
        "duration": page.text_content("#meta-duration"),
        "processing": page.text_content("#meta-processing"),
        "segments": page.eval_on_selector_all("#segment-list .segment", "els => els.length"),
    }
    print(f"      first partial after {first_partial_seconds}s: {partial_text}")
    print(f"      final: {final_text}")
    print(f"      segments: {results['streaming_zh']['segments']}")


def launch_browser(playwright, *, headless: bool, channel: str | None, extra_args: list[str]):
    options = {"headless": headless, "args": extra_args}
    if channel:
        options["channel"] = channel
    return playwright.chromium.launch(**options)


def scenario_permission_denied(playwright, base_url: str, results: dict, channel: str | None = None) -> None:
    print("[5/5] microphone permission denied ...")
    browser = launch_browser(
        playwright,
        headless=True,
        channel=channel,
        extra_args=[
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={SAMPLES / 'chinese.wav'}",
        ],
    )
    try:
        context = browser.new_context()
        page = context.new_page()
        page.goto(base_url + "/", wait_until="domcontentloaded")

        session = context.new_cdp_session(page)
        session.send(
            "Browser.setPermission",
            {
                "permission": {"name": "microphone"},
                "setting": "denied",
                "origin": base_url,
            },
        )

        page.click("#record-button")
        page.wait_for_selector("#alert-card", state="visible", timeout=20_000)
        title = page.text_content("#alert-title")
        detail = page.text_content("#alert-detail")
        page.screenshot(path=str(SCREENSHOTS / "ui-permission-denied.png"), full_page=True)

        crashed = page.evaluate("() => document.body === null")
        results["permission_denied"] = {
            "alert_title": title,
            "alert_detail": (detail or "")[:160],
            "page_alive": not crashed,
            "record_button_enabled_again": page.is_enabled("#record-button"),
        }
        print(f"      alert: {title}")
        context.close()
    finally:
        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Browser end-to-end check")
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument(
        "--channel",
        default=None,
        help="use an installed browser channel (msedge or chrome) instead of the bundled chromium",
    )
    parser.add_argument("--port", type=int, default=0, help="port for the test server (default: random)")
    parser.add_argument("--skip-streaming", action="store_true", help="skip the streaming scenario")
    parser.add_argument("--keep-server", action="store_true", help="leave the server running afterwards")
    parser.add_argument("--json-out", default=None, help="write the results as JSON to this path")
    args = parser.parse_args()

    if not (SAMPLES / "chinese.wav").is_file():
        print("samples/chinese.wav is missing; run scripts/prepare_samples.py first.", file=sys.stderr)
        return 2

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is not installed. Run:\n"
            "  python -m pip install -r requirements-dev.txt\n"
            "  python -m playwright install chromium",
            file=sys.stderr,
        )
        return 2

    port = args.port or free_port()
    base_url = f"http://127.0.0.1:{port}"
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)

    server, log_file = start_server(port)
    results: dict = {"base_url": base_url, "server_log": str(log_file)}
    console_errors: list[str] = []
    page_errors: list[str] = []

    try:
        wait_for_health(base_url)
        print(f"server ready at {base_url}")

        with sync_playwright() as playwright:
            browser = launch_browser(
                playwright,
                headless=not args.headed,
                channel=args.channel,
                extra_args=[
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                    f"--use-file-for-fake-audio-capture={SAMPLES / 'chinese.wav'}",
                    "--autoplay-policy=no-user-gesture-required",
                ],
            )
            context = browser.new_context(permissions=["microphone"])
            page = context.new_page()
            page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            try:
                scenario_model_ui(page, base_url, results)
                scenario_record_chinese(page, base_url, results)
                scenario_upload_english(page, base_url, results)
                if not args.skip_streaming:
                    scenario_streaming(page, base_url, results)
            finally:
                context.close()
                browser.close()

            scenario_permission_denied(playwright, base_url, results, channel=args.channel)

        results["console_errors"] = console_errors
        results["page_errors"] = page_errors

        checks = {
            "model_ui": len(results.get("model_ui", {}).get("options", [])) >= 5,
            "record_chinese": bool(results.get("record_zh", {}).get("has_chinese")),
            "record_assets": bool(results.get("record_zh", {}).get("player_visible")),
            "upload_english": "hello" in results.get("upload_en", {}).get("text", "").lower(),
            "srt_export": "-->" in results.get("upload_en", {}).get("srt_preview", ""),
            "streaming": bool(results.get("streaming_zh", {}).get("final_text", "").strip())
            if not args.skip_streaming
            else True,
            "permission_denied": "权限" in results.get("permission_denied", {}).get("alert_title", "")
            and results.get("permission_denied", {}).get("page_alive", False),
            "no_js_errors": not page_errors,
        }
        results["checks"] = checks

        print("\n=== summary ===")
        for name, passed in checks.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        print(f"  screenshots: {SCREENSHOTS}")
        print(f"  server log:  {log_file}")

        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  results:     {args.json_out}")

        return 0 if all(checks.values()) else 1

    except Exception as error:
        results["error"] = repr(error)
        print(f"\nE2E check failed: {error}", file=sys.stderr)
        try:
            log_tail = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print("\n--- server log (tail) ---", file=sys.stderr)
            print("\n".join(log_tail), file=sys.stderr)
        except OSError:
            pass
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    finally:
        if args.keep_server:
            print(f"server left running on {base_url} (pid {server.pid})")
        else:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    raise SystemExit(main())

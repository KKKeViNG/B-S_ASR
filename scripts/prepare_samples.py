"""Generate the bundled sample clips (Windows only, development helper).

The assessment asks for two ready-to-use samples (``samples/chinese.wav`` and
``samples/english.wav``) with a legal, distributable source. Instead of shipping
third-party recordings we synthesise them locally with the Windows SAPI voices
(Microsoft Huihui for zh-CN, Microsoft Zira for en-US) and normalise the result
to the 16 kHz mono PCM that the ASR pipeline expects.

Run once on a Windows machine that has the zh-CN and en-US speech voices::

    python scripts/prepare_samples.py

Both clips are generated on this machine, contain no personal data and can be
redistributed freely with the repository.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.audio import decode_audio, encode_wav  # noqa: E402

SAMPLES = [
    {
        "filename": "chinese.wav",
        "voice": "Microsoft Huihui Desktop",
        "rate": -1,
        "text": "你好，欢迎使用本地语音识别系统。今天天气很好，我们一起出去散步吧。",
    },
    {
        "filename": "english.wav",
        "voice": "Microsoft Zira Desktop",
        "rate": -1,
        "text": (
            "Hello and welcome to the local speech recognition system. "
            "The weather is nice today, so let's go for a walk."
        ),
    },
]

PS_SCRIPT = """
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("{voice}")
$synth.Rate = {rate}
$synth.Volume = 100
$text = Get-Content -Raw -Encoding UTF8 "{text_file}"
$synth.SetOutputToWaveFile("{out_file}")
$synth.Speak($text)
$synth.Dispose()
"""


def synthesise(text: str, voice: str, rate: int, destination: Path) -> None:
    """Ask Windows SAPI to render *text* into *destination* (any sample rate)."""

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        text_file = Path(handle.name)

    script = PS_SCRIPT.format(voice=voice, rate=rate, text_file=text_file, out_file=destination)
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", encoding="utf-8-sig", delete=False) as handle:
        handle.write(script)
        script_file = Path(handle.name)

    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_file)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        text_file.unlink(missing_ok=True)
        script_file.unlink(missing_ok=True)


def main() -> int:
    samples_dir = PROJECT_ROOT / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    reference_lines: list[str] = []

    for spec in SAMPLES:
        raw_path = samples_dir / f"_raw_{spec['filename']}"
        target_path = samples_dir / spec["filename"]
        print(f"Synthesising {spec['filename']} with '{spec['voice']}' ...")
        synthesise(spec["text"], spec["voice"], spec["rate"], raw_path)

        decoded = decode_audio(raw_path.read_bytes())
        target_path.write_bytes(encode_wav(decoded.samples, 16000))
        raw_path.unlink(missing_ok=True)

        normalised = decode_audio(target_path.read_bytes())
        print(
            f"  -> {target_path.name}: {normalised.duration_seconds:.2f}s, "
            f"{normalised.sample_rate} Hz, {normalised.samples.dtype}, "
            f"{target_path.stat().st_size} bytes"
        )
        reference_lines.append(f"{spec['filename']}\t{spec['text']}")

    (samples_dir / "reference.txt").write_text(
        "# Reference transcripts for the bundled audio samples.\n"
        "# Format: <file>\\t<human transcript>\n"
        "# Generated locally with Windows SAPI (zh-CN Huihui / en-US Zira).\n"
        + "\n".join(reference_lines)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {samples_dir / 'reference.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

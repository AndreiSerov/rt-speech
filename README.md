# rt-speech

A local live call copilot for macOS on Apple Silicon. It listens to system audio (the client) and the microphone (me), transcribes speech in near real time, and prints speaker-tagged turns. No audio leaves the machine.

This is an early MVP. See [the design](docs/mvp-design.md) for the milestones and decisions, and [the glossary](GLOSSARY.md) for what the terms mean.

## Status

| Milestone | Status |
|---|---|
| 1. Scaffold, config, `AudioSource`, `FileSource` | Done |
| 2. Segmenter and debug CLI | Done |
| 3. STT worker and turn aggregator | Done |
| 4. Swift capture and `MacCaptureSource` | In review |
| 5. `Suggester` stub | Not started |

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs the Python version the project pins.

```sh
uv sync --extra mlx
```

`mlx` installs the default speech engine, `mlx-whisper`, which runs on the Apple Silicon GPU. Use `--extra faster` for `faster-whisper`, or `--all-extras` for both. Plain `uv sync` is enough for the tests that run by default.

The first transcription downloads the model from Hugging Face, about 1.6 GB. Audio is never uploaded. After that, set `HF_HUB_OFFLINE=1` to stop the model version check at startup.

## Tests

```sh
uv run pytest            # everything that runs without audio hardware
uv run --all-extras pytest -m manual  # tests that need real models or hardware
```

## Configuration

Settings live in [`config.toml`](config.toml). Unknown keys and invalid values are errors, so a typo can't silently fall back to a default. Keep secrets out of this file; they belong in `.env`, which git ignores.

## Running each milestone

### Milestone 1

To replay a file from Python:

```python
import asyncio
from pathlib import Path

from rt_speech.audio import Stream
from rt_speech.file_source import FileSource


async def main() -> None:
    source = FileSource([(Path("call.wav"), Stream.SYSTEM)])
    async for stream, pcm, timestamp in source:
        print(stream, f"{timestamp:.3f}", pcm.shape)


asyncio.run(main())
```

`FileSource` accepts only 16 kHz mono 16-bit WAV files. Convert anything else with:

```sh
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le call.wav
```

On a Mac without `ffmpeg`, you can also generate test speech:

```sh
say -o call.wav --file-format=WAVE --data-format=LEI16@16000 "Hello, this is a test."
```

### Milestone 2

Dump each Segment of a WAV file as its own WAV, so you can listen to the cuts:

```sh
uv run python -m rt_speech.debug_segment call.wav --out segments/
```

Each run replaces any earlier `segment_*.wav` files in that directory, and leaves other files alone. To compare settings, change the `[segmenter]` values in `config.toml` and run it again.

Some tests generate speech with macOS `say`, including the Russian voice Milena.

### Milestone 3

Print speaker-tagged Turns from a WAV file:

```sh
uv run --extra mlx python -m rt_speech.transcribe call.wav --stream-label system
```

| Flag | Meaning |
|---|---|
| `--stream-label system\|mic` | Which Stream the file plays on: `system` is the Client, `mic` is Me. The default is `system`. |
| `--other other.wav` | A second file on the other Stream, to test two speakers. |
| `--language ru\|en\|auto` | Overrides every language in `config.toml`. |

Turns go to stdout and logs go to stderr, so add `2>/dev/null` to see only the Turns. Press Ctrl-C once to finish what was already heard, and again to quit at once.

To use `faster-whisper`, set `engine = "faster-whisper"` under `[stt]` in `config.toml` and install its extra.

### Milestone 4

Live capture needs the Swift CLI, which you build with the Xcode command line tools:

```sh
swift build -c release --package-path native/capture
```

Then print speaker-tagged Turns from a live call, running from the repo root:

```sh
uv run --extra mlx python -m rt_speech.live
```

`--language` and `--config` work as in milestone 3. System audio is the Client and the microphone is Me. Press Ctrl-C once to finish what was already heard, and again to quit at once. Use headphones, or the Client's words can also appear as Me.

Capture survives device changes such as connecting AirPods, and logs when a Stream stops delivering. macOS offers system audio only while a display is on, so capture keeps the display from idling off. The Client still isn't heard if you lock the screen or put the Mac to sleep.

#### Permissions

macOS grants these to the app you run the command from, such as your terminal, not to `rt-capture`:

| Permission | Where to allow it | Needed for |
|---|---|---|
| Screen & System Audio Recording | System Settings > Privacy & Security | System audio. No video is kept. |
| Microphone | System Settings > Privacy & Security | Your side of the call. |

macOS asks the first time. After allowing Screen Recording, quit and reopen the terminal app, because the permission only applies after a restart. A missing permission makes `live` exit with an error that names it.

To check capture without any permission, or to test one device on its own:

```sh
native/capture/.build/release/rt-capture --test-signal --duration 1 | wc -c  # 127596
native/capture/.build/release/rt-capture --no-mic > /dev/null     # system audio only; Ctrl-C to stop
native/capture/.build/release/rt-capture --no-system > /dev/null  # microphone only; Ctrl-C to stop
```

The Swift tests run with `swift test --package-path native/capture`, and the test that uses the real devices runs with `uv run pytest -m manual tests/test_mac_capture.py`.

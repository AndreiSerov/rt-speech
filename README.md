# rt-speech

A local live call copilot for macOS on Apple Silicon. It listens to system audio (the client) and the microphone (me), transcribes speech in near real time, and prints speaker-tagged turns. No audio leaves the machine.

This is an early MVP. See [the design](docs/mvp-design.md) for the milestones and decisions, and [the glossary](GLOSSARY.md) for what the terms mean.

## Status

| Milestone | Status |
|---|---|
| 1. Scaffold, config, `AudioSource`, `FileSource` | Done |
| 2. Segmenter and debug CLI | Done |
| 3. STT worker and turn aggregator | Not started |
| 4. Swift capture and `MacCaptureSource` | Not started |
| 5. `Suggester` stub | Not started |

## Setup

You need [uv](https://docs.astral.sh/uv/). It installs the Python version the project pins.

```sh
uv sync
```

## Tests

```sh
uv run pytest            # everything that runs without audio hardware
uv run pytest -m manual  # tests that need real hardware or permissions (none yet)
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

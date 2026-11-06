# rt-speech

A local live call copilot for macOS on Apple Silicon. It listens to system audio (the client) and the microphone (me), transcribes speech in near real time, prints speaker-tagged turns, and saves them to a text file with timecodes. After each client turn it prints a placeholder suggestion, ready for a real model later. No audio leaves the machine.

This is an early MVP. See [the design](docs/mvp-design.md) for the milestones and decisions, and [the glossary](GLOSSARY.md) for what the terms mean.

## Status

| Milestone | Status |
|---|---|
| 1. Scaffold, config, `AudioSource`, `FileSource` | Done |
| 2. Segmenter and debug CLI | Done |
| 3. STT worker and turn aggregator | Done |
| 4. Swift capture and `MacCaptureSource` | Done |
| 5. `Suggester` stub | Done |
| 6. Transcripts and MP3 | Done |

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

## Lint and format

```sh
uv run ruff check
uv run ruff format
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

`FileSource` accepts MP3 files of any sample rate and channel count, with no conversion, and WAV files that are 16 kHz mono 16-bit. Convert anything else with:

```sh
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le call.wav
```

On a Mac without `ffmpeg`, you can also generate test speech:

```sh
say -o call.wav --file-format=WAVE --data-format=LEI16@16000 "Hello, this is a test."
```

### Milestone 2

Dump each Segment of a WAV or MP3 file as its own WAV, so you can listen to the cuts:

```sh
uv run python -m rt_speech.debug_segment call.wav --out segments/
```

Each run replaces any earlier `segment_*.wav` files in that directory, and leaves other files alone. To compare settings, change the `[segmenter]` values in `config.toml` and run it again.

Some tests generate speech with macOS `say`, including the Russian voice Milena.

### Milestone 3

Print speaker-tagged Turns from a WAV or MP3 file:

```sh
uv run --extra mlx python -m rt_speech.transcribe call.mp3 --stream-label system
```

| Flag | Meaning |
|---|---|
| `--stream-label system\|mic` | Which Stream the file plays on: `system` is the Client, `mic` is Me. The default is `system`. |
| `--other other.mp3` | A second WAV or MP3 file on the other Stream, to test two speakers. |
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

Capture uses the macOS default input. Your speech transcribes much better from the MacBook microphone than from AirPods, so choose it under System Settings > Sound > Input. AirPods can stay as the output, but macOS may switch the input back when they reconnect.

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

### Milestone 5

`transcribe` and `live` now print a Suggestion each time a Client Turn closes. No model is involved yet: the text is a placeholder marked `(dummy)`, and nothing goes over the network.

```
[client] Можно ли получить скидку для небольшой команды?
[suggestion] (dummy) Follow up on "...ли получить скидку для небольшой команды?"
```

Settings are under `[suggester]` in `config.toml`:

| Setting | Meaning |
|---|---|
| `engine` | `dummy` prints placeholders, and `off` prints none. |
| `recent_turns` | How many Turns a Suggestion is based on. |
| `timeout_seconds` | How long to wait for a Suggestion before giving up. |

To add a real model, implement `Suggester` in `src/rt_speech/suggester.py`, add it to `make_suggester`, and add its name to `engine` in `config.py`.

### Milestone 6

`live` and `transcribe` now save a Transcript: a plain text file of the run's Turns. Each run prints its path to stderr when it starts. Suggestions aren't saved, and a run that hears no speech leaves no file.

| Command | File | First line |
|---|---|---|
| `live` | `transcripts/live/2026-11-06_14-30-05.txt` | `Встреча 2026-11-06 14:30:05` |
| `transcribe` | `transcripts/transcribe/call_2026-11-06_14-30-05.txt` | `Файл call.mp3, обработан 2026-11-06 14:30:05` |

For `live`, the date and time are when the meeting started. For `transcribe`, they're when the file was processed, not when the meeting happened. With `--other`, the first line lists both files and the name uses the main one.

```
Встреча 2026-11-06 14:30:05

[00:00:12] [client] Можно ли получить скидку?
[00:00:19] [me] Да, давайте обсудим.
```

Each timecode is when the Turn started, counted from the start of capture or of the file. Lines are in the order Turns closed, as in the terminal, so neighbouring timecodes can be slightly out of order.

Settings are under `[transcript]` in `config.toml`:

| Setting | Meaning |
|---|---|
| `enabled` | `false` saves nothing, for both commands. |
| `dir` | The root directory. A relative path starts from where you run the command. |

Transcripts hold private conversations as plain text:

- Git ignores `transcripts/`. If you set `dir` elsewhere, keep it out of git yourself.
- Don't point `dir` at a synced folder, such as `~/Documents` with iCloud Drive on, or Transcripts will leave the machine.
- Saving is on by default. Whether to record a call, and whether to tell the other side, is your decision.

Each Turn is written to disk as it closes, so quitting at once loses at most the Turn still open. A directory that can't be used stops the run before it starts. If saving fails during a run, such as on a full disk, you get one error, and Turns keep printing but are no longer saved.

A corrupt MP3 stops with an error, after saving what was heard before the damage. A file that is only cut short gives no error: you get a normal Transcript of the part that exists, so check the end of a recording you don't trust.

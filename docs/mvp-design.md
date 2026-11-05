# MVP design

The agreed scope and decisions for the live call copilot MVP: capture, segmentation, transcription and speaker-tagged turns printed in the terminal. The LLM suggestion stage is a stub.

## Constraints

- macOS on Apple Silicon only for now. Capture stays behind `AudioSource`, so a Windows backend (WASAPI loopback) can be added without touching the rest.
- Local-first: no audio leaves the machine in the MVP.
- Python 3.12 managed with uv, in the package `rt_speech`. A small Swift CLI handles capture.
- Calls may mix Russian and English, and both must be tested.

## Non-goals

Overlay UI, Windows support, per-app audio capture, live diarization, CRM or RAG context, cloud APIs, FastAPI, echo cancellation. RAG and hosted STT are wanted later.

## Architecture

```
Swift CLI (system + mic) ─┐
                          ├─> AudioSource -> segmenter (one per stream) -> STT worker -> turn aggregator -> terminal
WAV files (FileSource) ───┘
```

The pipeline depends only on `AudioSource`, an async iterator of `(stream, pcm, timestamp)`.

## Audio

- Frames are 512 samples (32 ms) of float32 mono audio at 16 kHz, because Silero VAD accepts only that size at 16 kHz.
- Timestamps are seconds since one session start shared by both streams.
- All timing logic uses these timestamps, never wall-clock time, so replays are fast and repeatable.
- `FileSource` takes one file per stream, merges them by timestamp, replays as fast as possible unless `realtime` is set, and rejects anything but 16 kHz mono 16-bit WAV.

## Segmenter

- State machine idle <-> speech. Enter speech after N consecutive voiced frames, and leave after about 600 ms of silence.
- About 300 ms of pre-roll, so the first syllable isn't clipped, and about 200 ms of post-roll after the last voiced frame. The rest of the closing silence is trimmed, so a Segment's end is close to when speech really stopped.
- Hard cap of 15–20 s per Segment: force a split at the quietest point of the last 2 s or so, not mid-word. Quietness is averaged over about 100 ms, since a shorter gap can be inside a word.
- Drop Segments with less than about 300 ms of speech, measured from the first voiced frame to the last, because pre-roll alone would otherwise satisfy the limit. The piece after a forced split is exempt, since it continues real speech.
- All thresholds are in config. Each Segment carries its Stream, start and end timestamps, and PCM.
- Voice detection uses Silero's ONNX model, committed to the repo and run through `onnxruntime`, behind a small interface. Each Stream has its own VAD state. The model is from silero-vad 6.2.3 (MIT), with its license alongside it. Tests compare the wrapper against reference scores from the official package.
- Measured on an M5 Pro with synthetic noise, single-threaded: 0.07 ms per frame, or 0.44% of one core for two Streams.

## Speech-to-text

- One transcription per Segment, in a worker, so capture never blocks. One worker serves both Streams in order.
- Segments are never dropped. A warning is logged when the queue passes a configurable limit.
- Engines sit behind a `Transcriber` interface: Segment audio and a language in, text with optional confidence values out. It must allow for network failures and timeouts, so a hosted engine can be added later.
- Each engine is an optional install (`uv sync --extra mlx`), because `mlx-whisper` requires torch. `mlx-whisper` is the default, `faster-whisper` is selectable, and the starting model is large-v3-turbo.
- Language is chosen at launch through config or `--language ru|en|auto`, with an optional per-Stream override. The default is `ru`. Previous text isn't passed as `initial_prompt` by default.
- The hallucination filter checks only the values an engine provides, with thresholds per engine. For Whisper, it drops text when `no_speech_prob` is high and `avg_logprob` is low. Known filler phrases on near-silence are checked for every engine. Each drop is logged with its reason.
- A future hosted engine must be selected explicitly and print a startup warning that audio is leaving the machine. Consider call-consent rules. Live-streaming APIs would need a different interface.

## Turns

- A Turn closes on any change of Speaker, or a pause of 1.5–2 s. A short "угу" therefore splits the other Speaker's Turn for now; milestone 3 will show how often that happens.
- Turns appear in the order their Segments finish, keeping real timestamps.
- "Pause reached" signals share the queue with Segments, so a Turn can't close while its last Segment is waiting for transcription.
- End of file or Ctrl-C finishes the open Segment, transcribes it, and closes all Turns.
- Emits `Turn(speaker, text, start, end)` events. The future LLM trigger subscribes to "Client Turn closed".

## Capture wire format

The Swift CLI writes logs to stderr, and fixed-size records to stdout, resampled to 16 kHz mono on the Swift side:

| Field | Size | Meaning |
|---|---|---|
| version | 1 byte | Format version |
| stream | 1 byte | 0 = system, 1 = mic |
| timestamp | 8 bytes | float64, seconds since session start |
| pcm | 2048 bytes | 512 × float32, little-endian |

Python reads this on a dedicated task that only moves frames, so slow STT can't stall capture.

## Milestones

Stop after each and report what works, what doesn't, and the main risks (latency, hallucinations, language flips).

1. Scaffold: uv project, package layout, config loading, `AudioSource`, `FileSource`, logging, pytest.
2. Segmenter and `python -m rt_speech.debug_segment input.wav --out segments/`, which dumps each Segment as a WAV. Unit tests use synthetic audio: silence, speech-like noise bursts, a long monologue.
3. STT worker and turn aggregator, printing speaker-tagged Turns from a WAV, with a `--stream-label` flag to treat the file as `system` or `mic`.
4. Swift capture CLI and `MacCaptureSource`, with both labeled Streams. Document the Screen Recording and Microphone permissions.
5. Stub `Suggester` interface (recent Turns in, short suggestions out) with a dummy implementation, wired to "Client Turn closed". No real LLM call.

## Working agreements

- Ask before any significant decision not covered here, such as library swaps or protocol changes.
- Small, focused commits, with one branch per milestone. The user merges.
- Test the segmenter and aggregator logic. Mark anything needing real audio hardware as `manual`.
- Keep the README current with setup, permissions, and how to run each milestone.

## Risks to measure

- STT speed and CPU/GPU load on real audio (milestone 3)
- Language flips with `auto` (milestone 3)
- Clock drift between the two audio devices on long calls (milestone 4)
- Without headphones, the Client's words can also appear as Me.
- Non-call audio on the computer counts as the Client.

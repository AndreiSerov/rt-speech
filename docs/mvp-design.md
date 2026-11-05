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
WAV files (FileSource) ───┘                                                                     |
                                                                                     each closed Client Turn
                                                                                                v
                                                                                            Suggester -> terminal
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
- The active engine gets the filter's thresholds too, so it can skip silence early. It drops such text itself, and the pipeline logs that it found no speech.
- A future hosted engine must be selected explicitly and print a startup warning that audio is leaving the machine. Consider call-consent rules. Live-streaming APIs would need a different interface.

## Turns

- A Turn closes on any change of Speaker, or a pause of 1.5–2 s. A short "угу" therefore splits the other Speaker's Turn for now; milestone 3 will show how often that happens.
- Turns appear in the order their Segments finish, keeping real timestamps.
- "Pause reached" signals share the queue with Segments, so a Turn can't close while its last Segment is waiting for transcription. A signal gives each Stream's own time, because one can lag behind the other, and says which Streams are mid-speech, because a Segment in progress may still continue the Turn.
- End of file or Ctrl-C finishes the open Segment, transcribes it, and closes all Turns.
- Emits `Turn(speaker, text, start, end)` events. Suggestions subscribe to "Client Turn closed".

## Suggestions

- A `Suggester` takes recent Turns, oldest first and ending with the Client Turn that just closed, and returns short Suggestions. The MVP's `DummySuggester` quotes the end of that Turn and marks its text `(dummy)`. It makes no network calls.
- The interface is async, because a real model will be a slow network call. Requests run in the background, so Turns keep printing meanwhile.
- A newer Client Turn cancels a Suggestion still in progress, since it answers something out of date. A Me Turn doesn't: it's added to the context of the next request.
- A Suggester failure or timeout is logged and the call goes on. When a run ends normally, including on Ctrl-C, the Suggestion in progress gets up to its timeout to finish. When a run fails, it's abandoned so the error shows at once, unless the Suggester ignores cancellation or is stuck in a thread.
- A real model would send the transcript off the machine, so like a hosted STT engine it must be selected explicitly and print a startup warning.

## Capture wire format

The Swift CLI writes logs to stderr, and fixed-size records to stdout, resampled to 16 kHz mono on the Swift side:

| Field | Size | Meaning |
|---|---|---|
| version | 1 byte | Format version |
| stream | 1 byte | 0 = system, 1 = mic |
| timestamp | 8 bytes | float64, seconds since session start |
| pcm | 2048 bytes | 512 × float32, little-endian |

Python reads this on a dedicated task that only moves frames, so slow STT can't stall capture.

## Keeping capture alive

A call outlasts any one device, so the Swift CLI stays up through device changes:

- A Stream with no audio for over 1 s gets silence, up to 0.5 s behind the clock, and a log line. Its clock keeps moving, so an open Segment or Turn can close. Gaps of any length are filled the same way when audio returns, so speech on either side stays apart.
- The microphone restarts on a new `AVAudioEngine` after a device change, retrying every second. The old engine stays silent after the change.
- System audio restarts its stream after a failure, retrying every 2 s. Stopping it from the macOS menu bar or losing the permission still ends the run.
- ScreenCaptureKit offers audio only through a display, so **system audio isn't heard while every display is off**. While capturing system audio, the CLI stops the display from idling off. Locking the screen or sleeping the Mac still stops system audio.

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

## Measured on an M5 Pro

These use speech synthesized by macOS `say`, so accuracy on real calls is still unknown.

| Engine | 3.4 s of speech takes |
|---|---|
| `mlx-whisper` large-v3-turbo | 0.19 s |
| `faster-whisper` large-v3-turbo, CPU int8 | 3.98 s |

- Russian, English and Russian with English words all transcribed correctly, with both `ru` and `auto` on the mixed file. Whisper spelled "deadline" and "feedback" in Cyrillic when the text was mainly Russian.
- **large-v3-turbo reports `no_speech_prob` 0.0 even on pure silence, and invents text with high confidence: "Продолжение следует..." in Russian, and "Thank you." in English.** The `no_speech_prob` rule therefore never fires on this model. large-v3 (0.61-0.84) and tiny (0.85-0.89) do report it.
- Silero is the real protection: 0 of 937 frames counted as voiced in each of five 30 s signals of silence, noise and hum. Filler phrases catch what remains on near-silence. Noise or music that Silero does accept could still produce invented text.

Live capture, playing a `say` phrase through the speakers at low volume:

- A Russian phrase came through word for word as a Client Turn 1.27 s after playback ended, including the pause that closes the Turn.
- ScreenCaptureKit keeps delivering frames during silence, so the system Stream's clock keeps advancing.
- Over 24 s neither Stream had a timestamp step above 32 ms.
- On real speech, 1.4-8.5 s Segments took 0.21-0.42 s to transcribe. System audio read well, including mixed Russian and English ("trade-off", "accuracy"). The microphone over AirPods, in their 24 kHz Bluetooth mode, got counting right but garbled the words after it, differently each time. The same phrase came through clearly on the MacBook microphone, in one try.
- An early version lost the microphone for 643.8 s across AirPods changes, and the whole capture when the display turned off. With the same changes, the current one restarted the microphone within a second each time, and system audio one second after the display came back. See "Keeping capture alive".

## Risks to measure

- STT speed and CPU/GPU load on real audio (milestone 3)
- Language flips with `auto` (milestone 3)
- Clock drift between the two audio devices on long calls (milestone 4)
- Without headphones, the Client's words can also appear as Me.
- macOS can make AirPods the default input whenever they connect, and speech from their microphone transcribes badly. Capture follows the default input, so choose the MacBook microphone in Sound settings.
- Non-call audio on the computer counts as the Client.
- A cap split landing just after speech ends can shorten that Segment's post-roll. It's rare, and it only removes audio after the last voiced frame.
- If the Python side stops reading, such as when the terminal is paused, the Swift CLI's write queue grows by about 650 MB an hour. Nothing is lost, and nothing limits it.
- A multi-channel interface with plain numbered channels is heard through its first channel only.
- A Client Turn closes on a pause, or once Me's first Segment is transcribed. After a quick reply, a Suggestion therefore starts only once Me has already spoken. Revisit this before adding a real model.
- Audio arriving over 1 s late overlaps the silence that filled its place, and that part is lost. Measured delivery delays stayed under 0.4 s.

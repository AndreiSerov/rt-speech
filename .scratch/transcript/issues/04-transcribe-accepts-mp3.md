# 04: `transcribe` accepts MP3

**What to build:** MP3 files of any sample rate and channel count work wherever `FileSource` is used, decoded as they play. See "MP3" in `.scratch/transcript/spec.md`.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] An MP3 replays as frames of the usual size at 16 kHz mono, for different source rates and for stereo, and its tail isn't lost.
- [x] The file extension is matched case-insensitively, and non-MP3 files keep today's WAV rules and errors.
- [x] A missing, corrupt or audio-free MP3 raises `FileSourceError` on construction, so it fails before an engine loads.
- [x] Damage partway through a file gives a clean error and exit code 1, with no traceback, after what was already heard is flushed.
- [x] `transcribe` runs an MP3 from end to end, and `--other` can mix an MP3 and a WAV across Streams.
- [x] `debug_segment` accepts an MP3.
- [x] Decoding doesn't load the whole file into memory.
- [x] PyAV is a required dependency, and MP3 tests build their files from synthetic audio, adding no audio files to the repo.

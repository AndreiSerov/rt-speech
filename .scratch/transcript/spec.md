# Save a Transcript of every run, and transcribe MP3 files

Status: ready-for-agent

## Problem Statement

Turns only appear in the terminal, so when a call ends, what was said is gone unless I copied it by hand. I can't look up what a Client said last week, or which meeting it came from, or where in the call they said it.

Replaying a recording has the same problem, and it only accepts 16 kHz mono WAV. Recordings usually arrive as MP3, so I have to convert each one with `ffmpeg` first.

## Solution

Every `live` and `transcribe` run saves a **Transcript**: a plain text file of its Turns, on this computer, named by when the run started. Each line carries a timecode counted from the start of capture or the start of the file. Runs of each command are kept in their own directory, inside a directory git ignores.

`transcribe` and the other users of `FileSource` also accept MP3 files of any sample rate and channel count, with no conversion step.

## User Stories

### Saving a call

1. As a user on a call, I want each Turn saved without doing anything, so that I can't forget to turn it on.
2. As a user, I want the file name to carry the date and time the call started, so that I can find a meeting by when it happened.
3. As a user with several meetings in a day, I want the time in the name down to the second, so that one meeting's Transcript never collides with another's.
4. As a user, I want a header stating the date and time of the meeting, so that the date survives if the file is renamed or moved.
5. As a user, I want each line to start with a timecode, so that I can tell how far into the call something was said.
6. As a user, I want timecodes counted from the start of capture, so that they match the same clock the pipeline uses and a quiet start to the call stays visible.
7. As a user, I want each line tagged with its Speaker the same way as the terminal, so that the file reads like what I saw on screen.
8. As a user, I want Turns in the order they closed, so that the file matches what the terminal showed, even when neighbouring timecodes are slightly out of order.
9. As a user, I want Suggestions left out, so that the Transcript holds only what was really said.
10. As a user, I want the Transcript's path printed when a run starts, so that I know where it will be.
11. As a user whose run heard no speech, I want no file left behind, so that the directory only holds real conversations.

### Keeping what was heard

12. As a user who presses Ctrl-C once, I want the open Turn finished and saved, so that the end of the call isn't lost.
13. As a user who presses Ctrl-C twice, I want every closed Turn already on disk, so that quitting at once loses at most the open Turn.
14. As a user whose capture fails mid-call, I want the Turns so far kept, so that a failure doesn't cost me the whole conversation.
15. As a user piping output into `head`, I want a Turn saved even when the terminal side has gone away, so that the Transcript doesn't depend on the reader of stdout.
16. As a user whose Transcript directory can't be created or written, I want a clear error before anything starts, so that I don't discover it after the call.
17. As a user whose disk fills up mid-call, I want one clear error and the Turns still printing, so that a save failure doesn't break the call I'm on.

### Transcribing files

18. As a user replaying a recording, I want its Transcript saved too, in its own directory, so that replays and real calls stay apart.
19. As a user replaying a recording, I want the name to carry the source file's name and the time of processing, so that I can tell which recording it came from.
20. As a user replaying a recording, I want the header to name the source file and say that the time is when it was processed, so that nobody mistakes it for the date of the meeting.
21. As a user replaying two files with `--other`, I want the header to list both and the name to use the main one, so that the source of both sides is on record.
22. As a user replaying a file, I want timecodes counted from the start of the file, so that I can find the moment in the recording.
23. As a user with an MP3, I want to pass it straight to `transcribe`, so that I can skip the `ffmpeg` step.
24. As a user with an MP3 at 44.1 kHz stereo, I want it accepted as is, so that I don't have to know how it was encoded.
25. As a user with a long recording, I want it decoded as it plays, so that memory use doesn't grow with its length.
26. As a user with a broken MP3, I want a clear error before any model loads, so that I don't wait for nothing.
27. As a user whose MP3 is damaged partway through, I want a clear error and exit code 1, with what was already heard kept, so that the first part of the recording still counts.
28. As a user with one MP3 and one WAV, I want to mix them across Streams, so that each side can come from whatever I have.
29. As a user cutting Segments with `debug_segment`, I want MP3 to work there too, so that I can tune the Segmenter on the same recordings.
30. As a user with a WAV that isn't 16 kHz mono 16-bit, I want the same error and hint as today, so that WAV behaviour doesn't change.

### Configuring

31. As a user, I want a setting that turns Transcripts off for both commands, so that I can run without saving anything.
32. As a user, I want a setting for the root directory, so that I can keep Transcripts wherever I choose.
33. As a user, I want a mistyped Transcript setting to be an error, so that it can't silently fall back to saving somewhere else.
34. As a user, I want the default directory ignored by git, so that private conversations can't be committed by accident.

### Developing

35. As a developer, I want Transcript as a term in the glossary, so that it isn't confused with the logs on stderr.
36. As a developer, I want the test suite to leave the real Transcript directory untouched, so that running tests never creates fake meetings.
37. As a developer, I want the README and design doc to cover Transcripts and MP3, so that they stay current as the working agreements require.

## Implementation Decisions

### Transcript

- A new module owns everything about a Transcript: the directory, the file name, the header, the line format, and handling of write failures. The commands only say which subdirectory to use, how to name the file, and what the header says, then hand it Turns.
- Add **Transcript** to the glossary: the Turns of one run, saved to disk. List _Log_ under its _Avoid_ terms, because logs are the stderr output.
- Add a `[transcript]` config section with `enabled` (default true) and `dir` (default `transcripts`). `enabled` covers both commands. A relative `dir` resolves from where the command runs, like `capture.binary`. Add it to the committed config file, and add the default directory to `.gitignore`.
- Each command writes to its own subdirectory of `dir`: `live` and `transcribe`.
- File names use the local time at the start of the run, formatted `YYYY-MM-DD_HH-MM-SS`, with a `.txt` extension. `live` uses just the time. `transcribe` prefixes the main input's stem and an underscore: `call_2026-11-06_14-30-05.txt`.
- Files are UTF-8. Each starts with a header line and a blank line:
  - `live`: `Встреча 2026-11-06 14:30:05`
  - `transcribe`: `Файл call.mp3, обработан 2026-11-06 14:30:05`, listing both files when `--other` is given
- Each Turn is one line: `[00:00:12] [client] text`. The timecode is the Turn's `start`, rounded down to whole seconds, as `HH:MM:SS`. Hours keep counting past 24. The Speaker tag is the terminal's.
- Pipeline timestamps already count from the session start, so no timestamp conversion is needed.
- Turns are written in closing order and never re-sorted. Only Turns are written.

### When things happen

- Before the engine loads and before capture starts, the root and subdirectory are created and checked with a real temporary write, since permission bits can disagree with what happens. A failure there prints `error: ...` and exits 1.
- The path is then printed to stderr, even though the file doesn't exist yet. Nothing is printed when `enabled` is false.
- The file is created by the first Turn, so a run without speech leaves no file. It never overwrites an existing file; a name collision counts as a write failure.
- Each Turn is flushed and synced to disk before the next step, because a second Ctrl-C exits through `os._exit` and skips all cleanup, and a power cut would lose what only reached the operating system. Turns are rare enough that the sync costs nothing noticeable.
- Each Turn is saved before it is printed, so a closed pipe can't lose it.
- After a write failure during a run, one error goes to stderr, nothing further is attempted for that run, Turns keep printing, and the exit code is unaffected. This failure must not reach the existing handler that turns an `OSError` into exit code 1.
- The local time is read when the command starts, just before capture. That small offset from timecode zero is acceptable.

### MP3

- `FileSource` chooses its reader by case-insensitive file extension: `.mp3` is decoded, and everything else keeps the existing WAV rules and errors. The `FileSource` contract doesn't change, so `transcribe`, `--other` and `debug_segment` get MP3 without their own changes.
- Decode with PyAV, added as a required dependency. It's already in the lockfile through `faster-whisper`.
- Convert any sample rate and channel count to 16 kHz mono as the file plays, without loading the whole file. Flush the resampler at the end so the tail isn't lost, and pad the last partial frame like WAV does.
- Frame timestamps come from the frame index, as for WAV. An encoder's short leading delay is accepted.
- Validation on construction opens the file, finds an audio stream and decodes the first audio, so missing, corrupt or audio-free MP3s raise `FileSourceError` before any engine loads.
- A decode failure during replay surfaces as a clean error with exit code 1, after what was already heard is flushed through the pipeline. PyAV's own exceptions must not reach the user as a traceback.

## Testing Decisions

- A good test drives a command or `FileSource` through its public interface and checks what the user sees: exit codes, stdout, stderr, frames, and the file on disk. It must not reach into the Transcript module or mock its internals. Expected values are literals.
- There are three seams, and no others:
  - **`transcribe.main(argv)`** with the fake engine, as `test_transcribe.py` does: location, name, header, line format, `--other`, `enabled = false`, no file without speech, startup and mid-run write failures, flushing per Turn, and one MP3 run from end to end.
  - **`live.main(argv)`** with a fake engine and a fake capture source, in a new test file using the same approach: the `live` subdirectory, the meeting header, and a startup failure before capture starts.
  - **`FileSource`**, as `test_file_source.py` does: MP3 to 16 kHz mono frames from different rates and channel counts, the tail is kept, corrupt and audio-free files fail on construction, a mid-file failure, uppercase extensions, and WAV and MP3 across Streams.
- Flushing is tested by having the fake engine read the Transcript on its next call and find the previous Turn. A mid-run write failure is tested by having the fake engine make the directory unwritable: one error, Turns keep printing, exit code 0.
- The start time is controlled in tests, so names and headers are checked exactly.
- MP3 test files are built in the tests with PyAV from synthetic audio. No audio files are added to the repo.
- **The test suite must not write to the real Transcript directory.** Every test runs from a scratch directory holding a copy of the committed config, and the run fails if the real directory changed.
- Config validation follows `test_config.py`.
- A real call through `live` stays a `manual` check for the user.

## Out of Scope

- Suggestions in the Transcript. Add them once a real model exists.
- Formats other than MP3, such as M4A, and more lenient WAV rules.
- Choosing a file name per run, or a CLI flag for the directory.
- Reading the date of a meeting from a recording's metadata.
- Retention, deletion, encryption or search of Transcripts.
- Retrying writes after a failure, or buffering Turns to save later.
- Sorting Turns by timecode.
- Any change to the Swift capture CLI.

## Further Notes

- Transcripts hold private conversations in plain text. The default location is inside the repo and ignored by git because a folder like `~/Documents` may sync to iCloud, which would break the promise that nothing leaves the machine. The README should say so, and warn against pointing `dir` at a synced folder.
- Saving is on by default. Whether to record a call and tell the other side remains the user's decision, and the README should say so.
- This is milestone 6, built on the `m6-transcript` branch.
- MP3 decoding is the isolated ticket for GLM 5.3 through `opencode`, as the workflow in `AGENTS.md` describes. Sonnet builds the rest.

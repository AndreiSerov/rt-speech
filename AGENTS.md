# rt-speech

Local live call copilot for macOS on Apple Silicon. The Python package is `rt_speech`; a small Swift CLI in `native/capture` handles audio capture.

- [`docs/mvp-design.md`](docs/mvp-design.md): read before changing pipeline behaviour, scope, or the capture wire format. Its "Working agreements" section covers when to ask first, branches and merging.
- [`GLOSSARY.md`](GLOSSARY.md): read before naming anything. Code, docs and logs use its terms (Stream, Speaker, Segment, Turn, Suggestion) and avoid the synonyms it lists.

## Done means

- `uv run pytest`, `uv run ruff check` and `uv run ruff format --check` all pass.
- After Swift changes, `swift test --package-path native/capture` passes too.
- Work needing real audio hardware, permissions or models is marked `manual`, and the user runs it. Report it as not run.

## Conventions

- Build test-first with the `tdd` skill.
- All audio timing uses frame timestamps rather than wall-clock time, which keeps replays fast and repeatable.
- Turns and Suggestions go to stdout, and logs go to stderr.
- A `# noqa` states its reason on the same or the previous line.
- Commit messages end at the body, with no `Co-Authored-By` line.

## Gotchas

- Audio and transcripts stay on the machine. Anything hosted must be selected explicitly and print a startup warning.
- Plain `uv sync` skips the speech engines, because `mlx-whisper` pulls in torch. Default tests don't need them. Real transcription needs `--extra mlx` or `--extra faster`.
- The capture wire format lives on both sides: change `native/capture` and `src/rt_speech/mac_capture.py` together.

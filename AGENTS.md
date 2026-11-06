# rt-speech

Local live call copilot for macOS on Apple Silicon. The Python package is `rt_speech`; a small Swift CLI in `native/capture` handles audio capture.

- [`docs/mvp-design.md`](docs/mvp-design.md): read before changing pipeline behaviour, scope, or the capture wire format. Its "Working agreements" section covers when to ask first, branches and merging.
- [`GLOSSARY.md`](GLOSSARY.md): read before naming anything. Code, docs and logs use its terms (Stream, Speaker, Segment, Turn, Suggestion) and avoid the synonyms it lists.

## Workflow

Features follow Matt Pocock's skills flow. `/ask-matt` routes anything that doesn't fit, such as upkeep; hard bugs start at `diagnosing-bugs`.

The user starts each `/`-prefixed step, because the agent can't invoke those skills.

1. **Grill**: `/grill-with-docs` asks about every decision until none is left open. New terms go in `GLOSSARY.md`, and lasting decisions in `docs/adr/`.
2. **Spec**: `/to-spec` turns the grilling into a spec, and the user confirms its test seams.
3. **Tickets**: `/to-tickets` splits a multi-session build into tracer-bullet tickets with blocking edges. Work that fits one session goes straight to step 4.
4. **Build**: `/implement` runs per ticket, with a cleared context between tickets, or `/implement-spec` runs the whole spec on one integration branch. Implementers drive `tdd`, one red-green slice at a time.
5. **Review**: `code-review` checks the diff against the standards and the spec, and every finding is fixed or reported.
6. **Ship**: push the feature branch and open a PR with `pr`. The user merges.
7. **Retro**: `/retro` runs before the session is cleared, to improve the environment rather than the code.

Keep steps 1-3 in one unbroken context, so the spec and tickets build on the grilling.

### Who does what

- The main session on the strongest model grills, specs, reviews and verifies the result.
- Cheaper agents implement: Sonnet by default, and GLM 5.3 through `opencode` (`opencode-go/glm-5.3`) for isolated tickets.
- External agents like GLM work in their own worktree, with project code and synthetic test data only. Keep call recordings, Transcripts and `.env` out of what they can read.
- Work from any model passes the same tests and review. A ticket that fails review goes back to Sonnet.

## Agent skills

### Issue tracker

Issues and specs live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

The default five role names, recorded on a `Status:` line in each issue. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Done means

- `uv run pytest`, `uv run ruff check` and `uv run ruff format --check` all pass.
- After Swift changes, `swift test --package-path native/capture` passes too.
- Work needing real audio hardware, permissions or models is marked `manual`, and the user runs it. Report it as not run.

## Conventions

- All audio timing uses frame timestamps rather than wall-clock time, which keeps replays fast and repeatable.
- Turns and Suggestions go to stdout, and logs go to stderr.
- A `# noqa` states its reason on the same or the previous line.
- Commit messages end at the body, with no `Co-Authored-By` line.

## Gotchas

- Audio and transcripts stay on the machine. Anything hosted must be selected explicitly and print a startup warning.
- Plain `uv sync` skips the speech engines, because `mlx-whisper` pulls in torch. Default tests don't need them. Real transcription needs `--extra mlx` or `--extra faster`.
- The capture wire format lives on both sides: change `native/capture` and `src/rt_speech/mac_capture.py` together.

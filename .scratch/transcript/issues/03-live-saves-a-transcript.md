# 03: `live` saves a Transcript

**What to build:** a `live` run saves its Turns to a Transcript dated by when the meeting started, kept apart from replays. See `.scratch/transcript/spec.md`.

**Blocked by:** 01, 02

**Status:** ready-for-agent

- [ ] With a fake capture source and fake engine, a run creates `YYYY-MM-DD_HH-MM-SS.txt` in the `live` subdirectory.
- [ ] The file starts with `Встреча YYYY-MM-DD HH:MM:SS` and a blank line, followed by Turns in the shared line format.
- [ ] Timecodes count from the start of capture, using the pipeline's timestamps.
- [ ] A directory that can't be used prints `error: ...` and exits 1 before capture starts.
- [ ] `[transcript] enabled = false` saves nothing.

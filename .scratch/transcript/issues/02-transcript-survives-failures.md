# 02: A Transcript survives failures

**What to build:** what was heard stays on disk through a forced quit, a closed pipe or a failed run, and a problem saving never breaks a call in progress. See "When things happen" in `.scratch/transcript/spec.md`.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] A directory that can't be created or written prints `error: ...` and exits 1 before the engine loads.
- [ ] Each Turn is flushed as soon as it closes: the fake engine's next call can read the previous Turn from the file.
- [ ] A Turn is saved before it is printed, so a closed pipe still leaves it in the file.
- [ ] Turns closed before a failed run stay in the file, and the run still reports its failure as before.
- [ ] A write failure during a run prints one error to stderr, attempts no further writes, keeps printing Turns, and leaves the exit code at 0.
- [ ] An existing file is never overwritten; a name collision counts as a write failure.

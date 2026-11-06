# 01: `transcribe` saves a Transcript

**What to build:** a `transcribe` run saves its Turns to a Transcript under the `transcribe` subdirectory of the configured root, as the spec describes. This is the tracer bullet: config, the Transcript module, and the wiring into the command. See `.scratch/transcript/spec.md`.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] A run on a WAV creates `<stem>_YYYY-MM-DD_HH-MM-SS.txt` in the `transcribe` subdirectory, named from the local start time.
- [ ] The file is UTF-8 and starts with `Файл <name>, обработан YYYY-MM-DD HH:MM:SS` and a blank line. With `--other`, the header lists both files and the name uses the main input.
- [ ] Each Turn is one line, `[HH:MM:SS] [speaker] text`, with its start rounded down to whole seconds, in closing order. Hours keep counting past 24. Suggestions are left out.
- [ ] The path is printed to stderr at the start of a run, and stdout is unchanged.
- [ ] A run with no speech leaves no file.
- [ ] `[transcript] enabled = false` saves nothing and prints no path.
- [ ] `[transcript] dir` sets the root, and an unknown key in the section is a config error.
- [ ] The committed config has the new section, and git ignores the default directory.
- [ ] Transcript is in the glossary, with Log under its _Avoid_ terms.
- [ ] A run of the test suite leaves no `transcripts/` directory in the repo.

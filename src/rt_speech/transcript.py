"""Saves the Turns of one run to a Transcript file."""

import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from rt_speech.config import TranscriptConfig
from rt_speech.turns import Turn

_FILE_STAMP = "%Y-%m-%d_%H-%M-%S"
_HEADER_STAMP = "%Y-%m-%d %H:%M:%S"


def _timecode(seconds: float) -> str:
    whole = max(int(seconds), 0)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


class TranscriptError(Exception):
    """The Transcript directory can't be created or written."""


class Transcript:
    """Created by the first Turn, so a run without speech leaves no file."""

    def __init__(self, path: Path, header: str) -> None:
        self._path = path
        self._header = header
        self._file: TextIO | None = None
        self._failed = False

    def save(self, turn: Turn) -> None:
        """Writes and flushes the Turn. After a failure, which is reported once, the
        Transcript is abandoned and the run goes on without it."""
        if self._failed:
            return
        try:
            if self._file is None:
                self._file = self._path.open("x", encoding="utf-8")
                self._file.write(f"{self._header}\n\n")
            line = f"[{_timecode(turn.start)}] [{turn.speaker}] {turn.text}\n"
            self._file.write(line)
            self._file.flush()
        except OSError as e:
            self._failed = True
            print(
                f"error: can't save the Transcript to {self._path}: {e}. "
                "Turns are still printed, but no more are saved.",
                file=sys.stderr,
            )

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._file.close()
        except OSError:
            pass  # Every Turn was flushed when it closed; nothing is left to lose.


def open_transcript(
    config: TranscriptConfig,
    *,
    subdir: str,
    name_prefix: str,
    header: Callable[[str], str],
    started: datetime,
) -> Transcript | None:
    """Returns None when Transcripts are off. `header` gets the start time as text."""
    if not config.enabled:
        return None
    directory = config.dir / subdir
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise TranscriptError(f"can't create {directory}: {e}") from e
    if not os.access(directory, os.W_OK | os.X_OK):
        raise TranscriptError(f"can't write to {directory}: permission denied")
    path = directory / f"{name_prefix}{started.strftime(_FILE_STAMP)}.txt"
    print(f"Transcript: {path}", file=sys.stderr)
    return Transcript(path, header(started.strftime(_HEADER_STAMP)))

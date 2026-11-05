"""Joins transcribed Segments into Turns."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from rt_speech.audio import Stream


class Speaker(StrEnum):
    CLIENT = "client"
    ME = "me"


_SPEAKER_OF_STREAM = {Stream.SYSTEM: Speaker.CLIENT, Stream.MIC: Speaker.ME}
_STREAM_OF_SPEAKER = {speaker: stream for stream, speaker in _SPEAKER_OF_STREAM.items()}


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    text: str
    start: float
    end: float


class TurnAggregator:
    """Holds at most one open Turn. Every method returns the Turns it closed."""

    def __init__(self, pause_seconds: float) -> None:
        self._pause = pause_seconds
        self._open: Turn | None = None

    def add(self, stream: Stream, text: str, start: float, end: float) -> list[Turn]:
        """Adds a transcribed Segment. Empty text changes nothing."""
        if not text:
            return []
        speaker = _SPEAKER_OF_STREAM[stream]
        current = self._open
        if (
            current is not None
            and current.speaker == speaker
            and start - current.end < self._pause
        ):
            joined = f"{current.text} {text}"
            self._open = Turn(
                speaker, joined, min(current.start, start), max(current.end, end)
            )
            return []
        self._open = Turn(speaker, text, start, end)
        return [current] if current is not None else []

    def clock(
        self, now: Mapping[Stream, float], speaking: frozenset[Stream]
    ) -> list[Turn]:
        """Closes the open Turn after a long enough pause on its Speaker's own Stream,
        unless that Stream is mid-speech, since that Segment may yet continue the Turn."""
        current = self._open
        if current is None:
            return []
        stream = _STREAM_OF_SPEAKER[current.speaker]
        if stream in speaking or stream not in now:
            return []
        if now[stream] - current.end < self._pause:
            return []
        return self.flush()

    def flush(self) -> list[Turn]:
        """Closes the open Turn, if any."""
        current, self._open = self._open, None
        return [current] if current is not None else []

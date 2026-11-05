"""Cuts one Stream's frames into Segments of speech."""

from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, AudioSource, Frame, Stream
from rt_speech.config import SegmenterConfig

_FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
_FRAME_MS = _FRAME_SECONDS * 1000


class VoiceDetector(Protocol):
    def speech_probability(self, pcm: NDArray[np.float32]) -> float: ...


@dataclass(frozen=True, eq=False)
class Segment:
    stream: Stream
    start: float
    end: float
    pcm: NDArray[np.float32]


def _to_frames(ms: int) -> int:
    return round(ms / _FRAME_MS)


class Segmenter:
    """State machine for a single Stream: idle <-> speech."""

    def __init__(
        self, stream: Stream, detector: VoiceDetector, config: SegmenterConfig
    ) -> None:
        self._stream = stream
        self._detector = detector
        self._threshold = config.voice_threshold
        self._start_frames = config.speech_start_frames
        self._silence_end_frames = max(_to_frames(config.silence_end_ms), 1)
        self._postroll_frames = _to_frames(config.postroll_ms)
        self._min_speech_frames = _to_frames(config.min_speech_ms)
        self._max_frames = max(_to_frames(config.max_segment_ms), 1)
        self._search_frames = max(_to_frames(config.split_search_ms), 1)
        self._window_frames = max(_to_frames(config.split_window_ms), 1)
        # While idle: the latest frames, ready to become pre-roll plus the start run.
        self._recent: deque[Frame] = deque(
            maxlen=_to_frames(config.preroll_ms) + self._start_frames
        )
        self._voiced_run = 0
        # While in speech: the open Segment's frames and which of them are voiced.
        self._frames: list[Frame] = []
        self._voiced: list[bool] = []
        self._last_voiced = 0
        # True when the open Segment continues one that hit the cap.
        self._continuation = False

    def feed(self, frame: Frame) -> list[Segment]:
        voiced = self._detector.speech_probability(frame.pcm) >= self._threshold
        if self._frames:
            return self._feed_speech(frame, voiced)
        self._feed_idle(frame, voiced)
        return []

    def flush(self) -> list[Segment]:
        """Closes any open Segment. Call this at the end of the Stream."""
        return self._close() if self._frames else []

    def _feed_idle(self, frame: Frame, voiced: bool) -> None:
        self._recent.append(frame)
        self._voiced_run = self._voiced_run + 1 if voiced else 0
        if self._voiced_run < self._start_frames:
            return
        self._frames = list(self._recent)
        speech_start = len(self._frames) - self._start_frames
        self._voiced = [False] * speech_start + [True] * self._start_frames
        self._last_voiced = len(self._frames) - 1
        self._recent.clear()
        self._voiced_run = 0

    def _feed_speech(self, frame: Frame, voiced: bool) -> list[Segment]:
        self._frames.append(frame)
        self._voiced.append(voiced)
        if voiced:
            self._last_voiced = len(self._frames) - 1
        silent_run = len(self._frames) - 1 - self._last_voiced
        if silent_run >= self._silence_end_frames:
            return self._close()
        if len(self._frames) >= self._max_frames:
            return self._split()
        return []

    def _close(self) -> list[Segment]:
        """Ends the open Segment. Frames past the post-roll stay available as pre-roll."""
        end = max(self._last_voiced + 1 + self._postroll_frames, 0)
        segments = self._emit(self._frames[:end], self._voiced[:end])
        self._recent.extend(self._frames[end:])
        self._frames, self._voiced = [], []
        self._continuation = False
        return segments

    def _split(self) -> list[Segment]:
        """Cuts at the quietest point near the cap. The rest stays open even if it's
        silent so far, because the pause hasn't yet reached silence_end_ms."""
        cut = self._quietest_cut()
        segments = self._emit(self._frames[:cut], self._voiced[:cut])
        self._frames, self._voiced = self._frames[cut:], self._voiced[cut:]
        self._last_voiced -= cut
        self._continuation = True
        return segments

    def _quietest_cut(self) -> int:
        search = self._frames[-self._search_frames :]
        energy = [float(np.sqrt(np.mean(frame.pcm**2))) for frame in search]
        window = min(self._window_frames, len(search))
        smoothed = np.convolve(energy, np.ones(window) / window, mode="valid")
        offset = len(self._frames) - len(search)
        cut = offset + int(np.argmin(smoothed)) + window // 2 + 1
        return min(cut, len(self._frames) - 1)  # Always leave a frame open.

    def _emit(self, frames: list[Frame], voiced: list[bool]) -> list[Segment]:
        if not any(voiced):
            return []
        # Measured on speech alone: pre-roll would make every Segment long enough.
        first = voiced.index(True)
        last = len(voiced) - 1 - voiced[::-1].index(True)
        if not self._continuation and last - first + 1 < self._min_speech_frames:
            return []
        return [
            Segment(
                self._stream,
                frames[0].timestamp,
                frames[-1].timestamp + _FRAME_SECONDS,
                np.concatenate([frame.pcm for frame in frames]),
            )
        ]


async def segment_stream(
    source: AudioSource,
    make_detector: Callable[[], VoiceDetector],
    config: SegmenterConfig,
) -> AsyncIterator[Segment]:
    """Segments every Stream in `source`, each with its own detector and state."""
    segmenters: dict[Stream, Segmenter] = {}
    # aclosing: stopping early must stop the source now, such as a capture process.
    async with aclosing(aiter(source)) as frames:
        async for frame in frames:
            segmenter = segmenters.get(frame.stream)
            if segmenter is None:
                segmenter = Segmenter(frame.stream, make_detector(), config)
                segmenters[frame.stream] = segmenter
            for segment in segmenter.feed(frame):
                yield segment
    for segmenter in segmenters.values():
        for segment in segmenter.flush():
            yield segment

"""The audio contract the whole pipeline depends on."""

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16_000
# Silero VAD accepts exactly this many samples per call at 16 kHz (32 ms).
FRAME_SAMPLES = 512


class Stream(StrEnum):
    SYSTEM = "system"
    MIC = "mic"


class Frame(NamedTuple):
    stream: Stream
    pcm: NDArray[np.float32]
    """Exactly FRAME_SAMPLES mono samples in [-1, 1]."""
    timestamp: float
    """Start of the frame, in seconds since a session start shared by all streams."""


class AudioSource(Protocol):
    """Anything that produces frames: a replayed file or live capture."""

    def __aiter__(self) -> AsyncIterator[Frame]: ...

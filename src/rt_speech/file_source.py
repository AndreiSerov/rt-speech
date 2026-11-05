"""Replays WAV files as an AudioSource, for development and tests."""

import asyncio
import heapq
import wave
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, Frame, Stream

_BYTES_PER_SAMPLE = 2


class FileSourceError(Exception):
    """A WAV file can't be replayed, or the inputs don't make sense together."""


def _conversion_hint(path: Path) -> str:
    return (
        f"Convert it with: ffmpeg -i {path} -ar {SAMPLE_RATE} -ac 1 "
        f"-c:a pcm_s16le converted.wav"
    )


def _validate(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
    except FileNotFoundError:
        raise FileSourceError(f"{path}: file not found") from None
    except (wave.Error, EOFError) as e:
        raise FileSourceError(
            f"{path}: not a readable PCM WAV ({e}). {_conversion_hint(path)}"
        ) from e

    if rate != SAMPLE_RATE or channels != 1 or width != _BYTES_PER_SAMPLE:
        raise FileSourceError(
            f"{path}: expected {SAMPLE_RATE} Hz mono 16-bit, got {rate} Hz, "
            f"{channels} channel(s), {width * 8}-bit. {_conversion_hint(path)}"
        )


def _read_frames(path: Path, stream: Stream) -> Iterator[Frame]:
    with wave.open(str(path), "rb") as wav:
        index = 0
        while data := wav.readframes(FRAME_SAMPLES):
            samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
            # Pad the last partial frame so its speech isn't lost.
            pcm = np.pad(samples, (0, FRAME_SAMPLES - samples.size))
            yield Frame(stream, pcm, index * FRAME_SAMPLES / SAMPLE_RATE)
            index += 1


class FileSource:
    """Merges one WAV per Stream by timestamp.

    Files are validated on construction, so bad input fails before any replay.
    Each iteration replays from the start. Frames with equal timestamps keep
    the order of `inputs`.
    """

    def __init__(
        self, inputs: list[tuple[Path, Stream]], *, realtime: bool = False
    ) -> None:
        if not inputs:
            raise FileSourceError("at least one input file is required")
        streams = [stream for _, stream in inputs]
        if len(set(streams)) != len(streams):
            raise FileSourceError(f"each stream can have only one file, got {streams}")
        for path, _ in inputs:
            _validate(path)
        self._inputs = inputs
        self._realtime = realtime

    async def __aiter__(self) -> AsyncIterator[Frame]:
        readers = [_read_frames(path, stream) for path, stream in self._inputs]
        merged = heapq.merge(*readers, key=lambda frame: frame.timestamp)
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            for frame in merged:
                # Always yield to the loop, so a fast replay can't starve other tasks.
                delay = started + frame.timestamp - loop.time() if self._realtime else 0
                await asyncio.sleep(max(delay, 0))
                yield frame
        finally:
            merged.close()
            for reader in readers:
                reader.close()

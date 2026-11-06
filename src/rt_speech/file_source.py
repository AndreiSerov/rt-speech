"""Replays WAV and MP3 files as an AudioSource, for development and tests."""

import asyncio
import heapq
import wave
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path

import av
import numpy as np

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, Frame, Stream

_BYTES_PER_SAMPLE = 2


class FileSourceError(Exception):
    """A file can't be replayed, or the inputs don't make sense together."""


def _conversion_hint(path: Path) -> str:
    return (
        f"Convert it with: ffmpeg -i {path} -ar {SAMPLE_RATE} -ac 1 "
        f"-c:a pcm_s16le converted.wav"
    )


def _is_mp3(path: Path) -> bool:
    return path.suffix.lower() == ".mp3"


def _validate(path: Path) -> None:
    if _is_mp3(path):
        _validate_mp3(path)
    else:
        _validate_wav(path)


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
    except FileNotFoundError:
        raise FileSourceError(f"{path}: file not found") from None
    except OSError as e:
        raise FileSourceError(f"{path}: {e}") from e
    except (wave.Error, EOFError) as e:
        raise FileSourceError(
            f"{path}: not a readable PCM WAV ({e}). {_conversion_hint(path)}"
        ) from e

    if rate != SAMPLE_RATE or channels != 1 or width != _BYTES_PER_SAMPLE:
        raise FileSourceError(
            f"{path}: expected {SAMPLE_RATE} Hz mono 16-bit, got {rate} Hz, "
            f"{channels} channel(s), {width * 8}-bit. {_conversion_hint(path)}"
        )


def _decode_mp3(path: Path) -> Iterator[np.ndarray]:
    """Yields the samples of an MP3 as 16 kHz mono float32 chunks, as it decodes."""
    try:
        with av.open(str(path)) as container:
            audio = container.streams.audio
            if not audio:
                raise FileSourceError(f"{path}: no audio found")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
            for decoded in container.decode(audio[0]):
                for resampled in resampler.resample(decoded):
                    yield resampled.to_ndarray().reshape(-1)
            # The resampler holds back the tail until it is flushed.
            for resampled in resampler.resample(None):
                yield resampled.to_ndarray().reshape(-1)
    except FileNotFoundError:
        raise FileSourceError(f"{path}: file not found") from None
    except (OSError, av.error.FFmpegError) as e:
        raise FileSourceError(f"{path}: can't decode this MP3 ({e})") from e


def _validate_mp3(path: Path) -> None:
    chunks = _decode_mp3(path)
    try:
        if next(chunks, None) is None:
            raise FileSourceError(f"{path}: no audio found")
    finally:
        chunks.close()


def _read_mp3_frames(path: Path, stream: Stream) -> Iterator[Frame]:
    pending = np.empty(0, dtype=np.float32)
    index = 0
    for chunk in _decode_mp3(path):
        pending = np.concatenate([pending, chunk])
        while pending.size >= FRAME_SAMPLES:
            pcm, pending = pending[:FRAME_SAMPLES], pending[FRAME_SAMPLES:]
            yield Frame(stream, pcm, index * FRAME_SAMPLES / SAMPLE_RATE)
            index += 1
    if pending.size:
        # Pad the last partial frame so its speech isn't lost.
        pcm = np.pad(pending, (0, FRAME_SAMPLES - pending.size))
        yield Frame(stream, pcm, index * FRAME_SAMPLES / SAMPLE_RATE)


def _read_frames(path: Path, stream: Stream) -> Iterator[Frame]:
    if _is_mp3(path):
        yield from _read_mp3_frames(path, stream)
    else:
        yield from _read_wav_frames(path, stream)


def _read_wav_frames(path: Path, stream: Stream) -> Iterator[Frame]:
    with wave.open(str(path), "rb") as wav:
        index = 0
        while data := wav.readframes(FRAME_SAMPLES):
            # A truncated file can end in half a sample.
            usable = len(data) - len(data) % _BYTES_PER_SAMPLE
            samples = np.frombuffer(data[:usable], dtype="<i2").astype(np.float32)
            samples /= 32768.0
            # Pad the last partial frame so its speech isn't lost.
            pcm = np.pad(samples, (0, FRAME_SAMPLES - samples.size))
            yield Frame(stream, pcm, index * FRAME_SAMPLES / SAMPLE_RATE)
            index += 1


class FileSource:
    """Merges one WAV or MP3 per Stream by timestamp.

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

    async def __aiter__(self) -> AsyncGenerator[Frame, None]:
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

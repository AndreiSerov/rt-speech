"""Live capture on macOS, through the Swift rt-capture process."""

import asyncio
import contextlib
import logging
import struct
from collections import deque
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np

from rt_speech.audio import FRAME_SAMPLES, Frame, Stream

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1
_HEADER = struct.Struct("<BBd")  # version, stream, timestamp
_PCM_BYTES = FRAME_SAMPLES * 4
_RECORD_BYTES = _HEADER.size + _PCM_BYTES
_STREAMS = {0: Stream.SYSTEM, 1: Stream.MIC}
_TERMINATE_SECONDS = 2.0
_STDERR_LINES_KEPT = 5


class CaptureError(OSError):
    """The capture process can't start, failed, or sent something unreadable."""


def _decode(record: bytes) -> Frame:
    version, stream_id, timestamp = _HEADER.unpack_from(record)
    if version != _FORMAT_VERSION:
        raise CaptureError(f"rt-capture sent format version {version}, expected 1")
    stream = _STREAMS.get(stream_id)
    if stream is None:
        raise CaptureError(f"rt-capture sent unknown stream {stream_id}")
    pcm = np.frombuffer(record, dtype="<f4", offset=_HEADER.size).astype(np.float32)
    return Frame(stream, pcm, timestamp)


class MacCaptureSource:
    """Spawns rt-capture and reads its labeled frames. The process stops when
    iteration ends for any reason."""

    def __init__(self, binary: Path, *args: str) -> None:
        self._binary = binary
        self._args = args

    async def __aiter__(self) -> AsyncGenerator[Frame, None]:
        try:
            process = await asyncio.create_subprocess_exec(
                self._binary,
                *self._args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Its own session, so Ctrl-C reaches only us, and it keeps capturing
                # while the pipeline finishes what was already heard.
                start_new_session=True,
            )
        except OSError as e:
            raise CaptureError(
                f"can't run {self._binary}: {e}. Build it with: "
                "swift build -c release --package-path native/capture"
            ) from e

        assert process.stdout is not None
        assert process.stderr is not None
        recent_errors: deque[str] = deque(maxlen=_STDERR_LINES_KEPT)
        stderr_task = asyncio.create_task(_log_lines(process.stderr, recent_errors))
        try:
            while True:
                try:
                    record = await process.stdout.readexactly(_RECORD_BYTES)
                except asyncio.IncompleteReadError as e:
                    await _finish_reading(process, stderr_task)
                    if process.returncode:
                        details = " ".join(recent_errors)
                        raise CaptureError(
                            f"rt-capture exited with {process.returncode}: {details}"
                        ) from None
                    if e.partial:
                        raise CaptureError(
                            f"rt-capture's output was cut off after {len(e.partial)} "
                            "bytes of a record"
                        ) from None
                    return
                yield _decode(record)
        finally:
            await _stop(process)
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task


async def _log_lines(stream: asyncio.StreamReader, recent: deque[str]) -> None:
    while line := await stream.readline():
        text = line.decode(errors="replace").strip()
        recent.append(text)
        logger.info("rt-capture: %s", text)


async def _finish_reading(
    process: asyncio.subprocess.Process, stderr_task: asyncio.Task[None]
) -> None:
    await process.wait()
    await stderr_task


async def _stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), _TERMINATE_SECONDS)
    except TimeoutError:
        process.kill()
        await process.wait()

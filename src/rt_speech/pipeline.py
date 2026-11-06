"""Runs audio through segmentation, transcription and turn aggregation."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing

from rt_speech.audio import SAMPLE_RATE, AudioSource, Frame, Stream
from rt_speech.config import Config, SttConfig
from rt_speech.segmenter import Segment, StreamClock, VoiceDetector, segment_events
from rt_speech.stt import HallucinationFilter, Transcriber, TranscriptPiece
from rt_speech.turns import Turn, TurnAggregator

logger = logging.getLogger(__name__)


def _language_for(stream: Stream, config: SttConfig) -> str | None:
    override = config.system_language if stream is Stream.SYSTEM else config.mic_language
    language = override or config.language
    return None if language == "auto" else language


def _filter_for_engine(config: SttConfig) -> HallucinationFilter:
    engine = (
        config.mlx_whisper if config.engine == "mlx-whisper" else config.faster_whisper
    )
    return HallucinationFilter(engine.filter)


async def _until_stopped(
    source: AudioSource, stop: asyncio.Event
) -> AsyncGenerator[Frame, None]:
    """Ends when `stop` is set, even while the source is waiting for its next frame.
    Frames already read still come through. One task reads the whole source, because
    a source may hold a timeout or a TaskGroup across its frames."""
    frames: asyncio.Queue[Frame | None] = asyncio.Queue()
    failure: BaseException | None = None

    async def read_source() -> None:
        nonlocal failure
        try:
            async with aclosing(aiter(source)) as reader:
                async for frame in reader:
                    frames.put_nowait(frame)
        except asyncio.CancelledError as e:
            if not stopping:
                failure = e
        except Exception as e:  # noqa: BLE001 - re-raised once the frames have drained
            failure = e
        finally:
            frames.put_nowait(None)

    async def stop_reader() -> None:
        nonlocal stopping
        await stop.wait()
        stopping = True
        reader_task.cancel()

    stopping = False
    reader_task = asyncio.create_task(read_source())
    stopper_task = asyncio.create_task(stop_reader())
    try:
        while (frame := await frames.get()) is not None:
            yield frame
    finally:
        stopping = True
        stopper_task.cancel()
        reader_task.cancel()
        await asyncio.gather(reader_task, stopper_task, return_exceptions=True)
    if failure is not None:
        raise failure


async def run_pipeline(
    source: AudioSource,
    make_detector: Callable[[], VoiceDetector],
    transcriber: Transcriber,
    config: Config,
    on_turn: Callable[[Turn], None],
    *,
    stop: asyncio.Event | None = None,
) -> None:
    """Runs until `source` ends or `stop` is set. Everything already heard is then
    transcribed and every Turn is closed, before any source failure is raised."""
    # Segments and clock signals share this queue, so a pause can't close a Turn while
    # its last Segment is still waiting for transcription.
    queue: asyncio.Queue[Segment | StreamClock | None] = asyncio.Queue()
    stop = stop or asyncio.Event()
    pending = _PendingCount(config.stt.queue_warning_segments)
    hallucination_filter = _filter_for_engine(config.stt)
    turns = TurnAggregator(config.turns.pause_ms / 1000)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    async def capture() -> None:
        try:
            events = segment_events(
                _until_stopped(source, stop), make_detector, config.segmenter
            )
            async with aclosing(events) as stream:
                async for event in stream:
                    if isinstance(event, Segment):
                        pending.add()
                    queue.put_nowait(event)
        finally:
            queue.put_nowait(None)

    async def transcribe(segment: Segment) -> str:
        language = _language_for(segment.stream, config.stt)
        started = time.monotonic()
        try:
            pieces: list[TranscriptPiece] = await asyncio.get_running_loop().run_in_executor(
                worker, transcriber.transcribe, segment.pcm, language
            )
        except Exception:
            logger.exception(
                "Transcription failed for %s %.2f-%.2f s",
                segment.stream,
                segment.start,
                segment.end,
            )
            return ""
        logger.info(
            "Transcribed %.2f s of %s audio in %.2f s%s",
            segment.pcm.size / SAMPLE_RATE,
            segment.stream,
            time.monotonic() - started,
            "" if pieces else ", and the engine found no speech",
        )
        return hallucination_filter.keep(pieces, segment.pcm)

    async def consume() -> None:
        while (event := await queue.get()) is not None:
            if isinstance(event, StreamClock):
                closed = turns.clock(event.now, event.speaking)
            else:
                text = await transcribe(event)
                pending.done()
                closed = turns.add(event.stream, text, event.start, event.end)
            for turn in closed:
                on_turn(turn)
        for turn in turns.flush():
            on_turn(turn)

    capture_task = asyncio.create_task(capture())
    try:
        await consume()
    finally:
        capture_task.cancel()
        worker.shutdown(wait=False)
        # Raises the source's failure, if that's why the queue ended.
        results = await asyncio.gather(capture_task, return_exceptions=True)
    if isinstance(results[0], BaseException):
        raise results[0]


class _PendingCount:
    """Tracks Segments awaiting transcription, and warns once per backlog."""

    def __init__(self, warn_above: int) -> None:
        self._warn_above = warn_above
        self._count = 0
        self._warned = False

    def add(self) -> None:
        self._count += 1
        if self._count > self._warn_above and not self._warned:
            self._warned = True
            logger.warning(
                "Transcription is falling behind: %d Segments waiting", self._count
            )

    def done(self) -> None:
        self._count -= 1
        if self._count <= self._warn_above:
            self._warned = False

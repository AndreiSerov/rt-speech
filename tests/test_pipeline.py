import asyncio
import logging
import threading
from collections.abc import AsyncGenerator, Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, Frame, Stream
from rt_speech.config import Config, SegmenterConfig, SttConfig, TurnsConfig
from rt_speech.pipeline import run_pipeline
from rt_speech.stt import TranscriptPiece
from rt_speech.turns import Speaker, Turn

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
LOUD = 0.5
QUIET = 0.0
TIMEOUT = 5.0

CONFIG = Config(
    segmenter=SegmenterConfig(
        speech_start_frames=3,
        silence_end_ms=320,  # 10 frames
        preroll_ms=0,
        postroll_ms=0,
        min_speech_ms=160,  # 5 frames
        max_segment_ms=3200,
        split_search_ms=640,
    ),
    turns=TurnsConfig(pause_ms=640),  # 20 frames
)

Levels = dict[Stream, list[float]]


class LoudnessDetector:
    def speech_probability(self, pcm: NDArray[np.float32]) -> float:
        return 1.0 if np.abs(pcm).max() > 0.1 else 0.0


class Script:
    """A source that plays frame levels, then runs an optional step before ending."""

    def __init__(self, levels: Levels, before_end: Callable[[], object] | None = None):
        self.levels = levels
        self.before_end = before_end
        self.closed = False

    async def __aiter__(self) -> AsyncGenerator[Frame, None]:
        try:
            for index in range(max(len(v) for v in self.levels.values())):
                for stream, levels in self.levels.items():
                    if index < len(levels):
                        pcm = np.full(FRAME_SAMPLES, levels[index], dtype=np.float32)
                        yield Frame(stream, pcm, index * FRAME_SECONDS)
                await asyncio.sleep(0)
            if self.before_end:
                result = self.before_end()
                if asyncio.iscoroutine(result):
                    await result
        finally:
            self.closed = True


class FakeTranscriber:
    """Numbers each Segment, or follows a custom step before answering."""

    def __init__(self, step: Callable[[int], None] | None = None) -> None:
        self.step = step
        self.languages: list[str | None] = []
        self.threads: set[int] = set()

    def transcribe(
        self, pcm: NDArray[np.float32], language: str | None
    ) -> list[TranscriptPiece]:
        self.languages.append(language)
        self.threads.add(threading.get_ident())
        number = len(self.languages)
        if self.step:
            self.step(number)
        return [TranscriptPiece(f"s{number}")]


def run(
    source: Script,
    transcriber: FakeTranscriber,
    config: Config = CONFIG,
    stop: asyncio.Event | None = None,
) -> list[Turn]:
    turns: list[Turn] = []

    async def main() -> None:
        await asyncio.wait_for(
            run_pipeline(
                source,
                LoudnessDetector,
                transcriber,
                config,
                turns.append,
                stop=stop,
            ),
            TIMEOUT,
        )

    asyncio.run(main())
    return turns


def test_each_stream_becomes_its_speakers_turn() -> None:
    levels = {
        Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 80,
        Stream.MIC: [QUIET] * 40 + [LOUD] * 20 + [QUIET] * 40,
    }

    turns = run(Script(levels), FakeTranscriber())

    assert [(t.speaker, t.text) for t in turns] == [
        (Speaker.CLIENT, "s1"),
        (Speaker.ME, "s2"),
    ]


def test_segments_join_a_turn_across_a_short_pause() -> None:
    levels = {Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 12 + [LOUD] * 20 + [QUIET] * 40}

    turns = run(Script(levels), FakeTranscriber())

    assert [t.text for t in turns] == ["s1 s2"]


def test_a_pause_closes_the_turn_while_the_stream_keeps_running() -> None:
    turns: list[Turn] = []

    async def check_turn_already_closed() -> None:
        for _ in range(100):
            if turns:
                return
            await asyncio.sleep(0.01)
        raise AssertionError("turn did not close during the silence")

    source = Script(
        {Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 60}, check_turn_already_closed
    )

    async def main() -> None:
        await run_pipeline(
            source, LoudnessDetector, FakeTranscriber(), CONFIG, turns.append
        )

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))
    assert len(turns) == 1


def test_slow_transcription_never_blocks_capture() -> None:
    capture_done = threading.Event()
    held = threading.Event()

    def wait_for_capture(_: int) -> None:
        held.set()
        assert capture_done.wait(TIMEOUT), "capture was blocked by transcription"

    source = Script(
        {Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 40 + [LOUD] * 20 + [QUIET] * 40},
        capture_done.set,
    )

    assert len(run(source, FakeTranscriber(wait_for_capture))) == 2
    assert held.is_set()


def test_a_turn_does_not_close_while_its_next_segment_awaits_transcription() -> None:
    # Capture races far ahead through the silence while the first Segment is being
    # transcribed. Clock signals must wait their turn behind the second Segment.
    capture_done = threading.Event()

    def wait_for_capture(number: int) -> None:
        if number == 1:
            assert capture_done.wait(TIMEOUT)

    levels = {Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 12 + [LOUD] * 20 + [QUIET] * 200}

    turns = run(Script(levels, capture_done.set), FakeTranscriber(wait_for_capture))

    assert [t.text for t in turns] == ["s1 s2"]


def test_a_turn_stays_open_while_its_stream_is_mid_speech() -> None:
    # The second Segment starts within the pause but ends long after it would expire.
    levels = {Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 15 + [LOUD] * 60 + [QUIET] * 40}

    assert [t.text for t in run(Script(levels), FakeTranscriber())] == ["s1 s2"]


def test_a_failed_transcription_is_logged_and_the_rest_continue(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_first(number: int) -> None:
        if number == 1:
            raise RuntimeError("engine fell over")

    levels = {Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 40 + [LOUD] * 20 + [QUIET] * 40}

    with caplog.at_level(logging.ERROR):
        turns = run(Script(levels), FakeTranscriber(fail_first))

    assert [t.text for t in turns] == ["s2"]
    assert "engine fell over" in caplog.text


def test_a_growing_queue_logs_a_warning_and_drops_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    capture_done = threading.Event()

    def wait_for_capture(number: int) -> None:
        if number == 1:
            assert capture_done.wait(TIMEOUT)

    config = CONFIG.model_copy(
        update={"stt": SttConfig(queue_warning_segments=2)}
    )
    speech = [LOUD] * 20 + [QUIET] * 40
    source = Script({Stream.SYSTEM: speech * 4}, capture_done.set)

    with caplog.at_level(logging.WARNING):
        turns = run(source, FakeTranscriber(wait_for_capture), config)

    assert len(turns) == 4
    assert "behind" in caplog.text


def test_each_stream_uses_its_language() -> None:
    config = CONFIG.model_copy(
        update={"stt": SttConfig(language="ru", system_language="auto", mic_language="en")}
    )
    levels = {
        Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 80,
        Stream.MIC: [QUIET] * 40 + [LOUD] * 20 + [QUIET] * 40,
    }
    transcriber = FakeTranscriber()

    run(Script(levels), transcriber, config)

    assert transcriber.languages == [None, "en"]


def test_the_default_language_applies_without_an_override() -> None:
    config = CONFIG.model_copy(update={"stt": SttConfig(language="ru")})
    transcriber = FakeTranscriber()

    run(Script({Stream.MIC: [LOUD] * 20 + [QUIET] * 40}), transcriber, config)

    assert transcriber.languages == ["ru"]


def test_all_transcription_happens_on_one_worker_thread() -> None:
    levels = {Stream.SYSTEM: ([LOUD] * 20 + [QUIET] * 40) * 3}
    transcriber = FakeTranscriber()

    run(Script(levels), transcriber)

    assert len(transcriber.threads) == 1
    assert threading.get_ident() not in transcriber.threads


def test_filtered_text_makes_no_turn() -> None:
    class Hallucinating(FakeTranscriber):
        def transcribe(
            self, pcm: NDArray[np.float32], language: str | None
        ) -> list[TranscriptPiece]:
            return [TranscriptPiece("Шум", no_speech_prob=0.99, avg_logprob=-5.0)]

    source = Script({Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 40})

    assert run(source, Hallucinating()) == []


def test_stop_flushes_the_open_segment_closes_turns_and_closes_the_source() -> None:
    stop_holder: dict[str, asyncio.Event] = {}
    turns: list[Turn] = []

    class StopsMidSpeech(Script):
        async def __aiter__(self) -> AsyncGenerator[Frame, None]:
            index = 0
            try:
                while True:
                    pcm = np.full(FRAME_SAMPLES, LOUD, dtype=np.float32)
                    yield Frame(Stream.MIC, pcm, index * FRAME_SECONDS)
                    index += 1
                    if index == 20:
                        stop_holder["stop"].set()
                    await asyncio.sleep(0)
            finally:
                self.closed = True

    source = StopsMidSpeech({})

    async def main() -> None:
        stop_holder["stop"] = asyncio.Event()
        await run_pipeline(
            source,
            LoudnessDetector,
            FakeTranscriber(),
            CONFIG,
            turns.append,
            stop=stop_holder["stop"],
        )

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert [(t.speaker, t.text) for t in turns] == [(Speaker.ME, "s1")]
    assert source.closed


def test_a_source_failure_still_delivers_finished_speech_then_raises() -> None:
    class Failing(Script):
        async def __aiter__(self) -> AsyncGenerator[Frame, None]:
            for index in range(20):
                pcm = np.full(FRAME_SAMPLES, LOUD, dtype=np.float32)
                yield Frame(Stream.SYSTEM, pcm, index * FRAME_SECONDS)
            raise OSError("capture died")

    turns: list[Turn] = []

    async def main() -> None:
        await run_pipeline(
            Failing({}), LoudnessDetector, FakeTranscriber(), CONFIG, turns.append
        )

    with pytest.raises(OSError, match="capture died"):
        asyncio.run(asyncio.wait_for(main(), TIMEOUT))
    assert [t.text for t in turns] == ["s1"]


def test_a_failing_turn_callback_stops_the_pipeline_without_hanging() -> None:
    def explode(_: Turn) -> None:
        raise ValueError("subscriber broke")

    source = Script({Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 200})

    async def main() -> None:
        await run_pipeline(
            source, LoudnessDetector, FakeTranscriber(), CONFIG, explode
        )

    with pytest.raises(ValueError, match="subscriber broke"):
        asyncio.run(asyncio.wait_for(main(), TIMEOUT))
    assert source.closed

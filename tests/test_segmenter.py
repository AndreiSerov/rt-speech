import asyncio
import random
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
import pytest
from numpy.typing import NDArray

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, Frame, Stream
from rt_speech.config import SegmenterConfig
from rt_speech.segmenter import Segment, Segmenter, segment_stream

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


class LoudnessDetector:
    """Calls any audible frame speech, so tests can script exact voiced frames."""

    def speech_probability(self, pcm: NDArray[np.float32]) -> float:
        return 1.0 if np.abs(pcm).max() > 0.1 else 0.0


CONFIG = SegmenterConfig(
    voice_threshold=0.5,
    speech_start_frames=3,
    silence_end_ms=320,  # 10 frames
    preroll_ms=160,  # 5 frames
    postroll_ms=64,  # 2 frames
    min_speech_ms=160,  # 5 frames
    max_segment_ms=3200,  # 100 frames
    split_search_ms=640,  # 20 frames
    split_window_ms=96,  # 3 frames
)

LOUD = 0.5
QUIET = 0.0


def run(
    levels: list[float], config: SegmenterConfig = CONFIG, *, flush: bool = True
) -> list[Segment]:
    """Feeds one frame per level, each holding that constant amplitude."""
    segmenter = Segmenter(Stream.MIC, LoudnessDetector(), config)
    segments: list[Segment] = []
    for index, level in enumerate(levels):
        pcm = np.full(FRAME_SAMPLES, level, dtype=np.float32)
        segments += segmenter.feed(Frame(Stream.MIC, pcm, index * FRAME_SECONDS))
    if flush:
        segments += segmenter.flush()
    return segments


def test_silence_produces_no_segments() -> None:
    assert run([QUIET] * 200) == []


def test_speech_followed_by_silence_produces_one_segment() -> None:
    [segment] = run([QUIET] * 20 + [LOUD] * 20 + [QUIET] * 20, flush=False)

    assert segment.stream is Stream.MIC
    assert segment.pcm.dtype == np.float32
    assert segment.pcm.size == round((segment.end - segment.start) * SAMPLE_RATE)


def test_preroll_is_kept_before_the_first_voiced_frame() -> None:
    # Voiced from frame 20; 5 frames of pre-roll reach back to frame 15.
    [segment] = run([QUIET] * 20 + [LOUD] * 20 + [QUIET] * 20, flush=False)

    assert segment.start == 15 * FRAME_SECONDS


def test_preroll_is_shorter_when_speech_starts_at_the_beginning() -> None:
    [segment] = run([LOUD] * 20 + [QUIET] * 20, flush=False)

    assert segment.start == 0.0


def test_segment_ends_postroll_after_the_last_voiced_frame() -> None:
    # Voiced frames are 20..39, so the end is frame 40 plus 2 frames of post-roll.
    [segment] = run([QUIET] * 20 + [LOUD] * 20 + [QUIET] * 20, flush=False)

    assert segment.end == 42 * FRAME_SECONDS


def test_voiced_blips_shorter_than_the_start_run_are_ignored() -> None:
    assert run(([LOUD] * 2 + [QUIET] * 2) * 30) == []


def test_silence_shorter_than_the_end_threshold_stays_in_the_segment() -> None:
    # A 9-frame pause is one frame short of closing.
    levels = [LOUD] * 10 + [QUIET] * 9 + [LOUD] * 10 + [QUIET] * 20

    assert len(run(levels, flush=False)) == 1


def test_a_long_enough_silence_splits_two_segments() -> None:
    levels = [LOUD] * 10 + [QUIET] * 10 + [LOUD] * 10 + [QUIET] * 20

    assert len(run(levels, flush=False)) == 2


def test_consecutive_segments_never_share_audio() -> None:
    # The gap is exactly silence_end_ms, the smallest that splits, so pre-roll and
    # post-roll compete for the same frames.
    config = CONFIG.model_copy(update={"preroll_ms": 320, "postroll_ms": 320})
    levels = [LOUD] * 10 + [QUIET] * 10 + [LOUD] * 10 + [QUIET] * 20

    first, second = run(levels, config, flush=False)

    assert first.end <= second.start


def test_speech_too_short_is_dropped_even_though_preroll_pads_it() -> None:
    # 4 voiced frames is under min_speech_ms (5), although with pre-roll and
    # post-roll the audio would be 11 frames long.
    assert run([QUIET] * 20 + [LOUD] * 4 + [QUIET] * 20, flush=False) == []


def test_speech_exactly_the_minimum_is_kept() -> None:
    assert len(run([QUIET] * 20 + [LOUD] * 5 + [QUIET] * 20, flush=False)) == 1


def test_flush_closes_the_open_segment_at_the_end_of_the_stream() -> None:
    [segment] = run([QUIET] * 10 + [LOUD] * 10)

    assert segment.end == 20 * FRAME_SECONDS


def test_flush_applies_the_minimum_length() -> None:
    assert run([QUIET] * 10 + [LOUD] * 4) == []


def test_flush_with_nothing_open_produces_nothing() -> None:
    assert run([QUIET] * 10) == []


def test_flush_trims_silence_after_the_postroll() -> None:
    [segment] = run([LOUD] * 10 + [QUIET] * 5)

    assert segment.end == 12 * FRAME_SECONDS


STRONG = 0.9
WEAK = 0.2  # Still voiced, but much quieter.


def test_a_long_monologue_is_split_at_the_cap_without_losing_audio() -> None:
    segments = run([LOUD] * 250)

    assert len(segments) >= 3
    assert all(s.pcm.size <= 100 * FRAME_SAMPLES for s in segments)
    assert segments[0].start == 0.0
    assert segments[-1].end == 250 * FRAME_SECONDS
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert earlier.end == pytest.approx(later.start)
    assert sum(s.pcm.size for s in segments) == 250 * FRAME_SAMPLES


def test_the_split_lands_on_the_quietest_point_of_the_search_window() -> None:
    # The search window is frames 80..99, and the dip is frames 85..87.
    levels = [STRONG] * 85 + [WEAK] * 3 + [STRONG] * 60

    first, *_ = run(levels)

    assert 85 * FRAME_SECONDS < first.end <= 88 * FRAME_SECONDS


def test_a_dip_before_the_search_window_is_ignored() -> None:
    levels = [STRONG] * 50 + [WEAK] * 3 + [STRONG] * 100

    first, *_ = run(levels)

    assert first.end > 80 * FRAME_SECONDS


def test_one_quiet_frame_loses_to_a_wider_dip() -> None:
    # A single quiet frame can be a closure inside a word; a wider dip is a real gap.
    levels = [STRONG] * 82 + [WEAK] * 3 + [STRONG] * 8 + [0.11] + [STRONG] * 60

    first, *_ = run(levels)

    assert 82 * FRAME_SECONDS < first.end <= 85 * FRAME_SECONDS


def test_the_piece_after_a_split_can_be_shorter_than_the_minimum() -> None:
    # After the cut at frames 95..97, only a few voiced frames remain. They continue
    # real speech, so losing them would lose the end of a word.
    levels = [STRONG] * 95 + [WEAK] * 3 + [STRONG] * 3 + [QUIET] * 20

    segments = run(levels, flush=False)

    assert len(segments) == 2
    assert segments[1].start == pytest.approx(segments[0].end)


def test_silence_after_a_split_closes_the_segment_normally() -> None:
    levels = [LOUD] * 110 + [QUIET] * 20

    segments = run(levels, flush=False)

    assert segments[-1].end == 112 * FRAME_SECONDS


def test_a_split_inside_a_pause_returns_to_idle_when_no_speech_follows() -> None:
    # The cap is reached during a pause shorter than silence_end_ms.
    levels = [LOUD] * 92 + [QUIET] * 8 + [QUIET] * 20 + [LOUD] * 10 + [QUIET] * 20

    segments = run(levels, flush=False)

    assert len(segments) == 2
    assert segments[1].start == (120 - 5) * FRAME_SECONDS


def test_a_new_segment_after_a_split_uses_the_minimum_again() -> None:
    levels = [LOUD] * 110 + [QUIET] * 20 + [LOUD] * 4 + [QUIET] * 20

    assert all(s.end <= 112 * FRAME_SECONDS for s in run(levels, flush=False))


class ScriptedSource:
    """Two streams speaking at different times."""

    def __init__(self, levels: dict[Stream, list[float]]) -> None:
        self._levels = levels

    async def __aiter__(self) -> AsyncIterator[Frame]:
        for index in range(max(len(v) for v in self._levels.values())):
            for stream, levels in self._levels.items():
                if index < len(levels):
                    pcm = np.full(FRAME_SAMPLES, levels[index], dtype=np.float32)
                    yield Frame(stream, pcm, index * FRAME_SECONDS)


def collect_segments(levels: dict[Stream, list[float]]) -> list[Segment]:
    async def run_all() -> list[Segment]:
        stream = segment_stream(ScriptedSource(levels), LoudnessDetector, CONFIG)
        return [segment async for segment in stream]

    return asyncio.run(run_all())


def test_each_stream_is_segmented_independently() -> None:
    # Speech on the mic must not keep the system Segment open, and vice versa.
    segments = collect_segments(
        {
            Stream.SYSTEM: [LOUD] * 20 + [QUIET] * 60,
            Stream.MIC: [QUIET] * 10 + [LOUD] * 40 + [QUIET] * 30,
        }
    )

    assert [(s.stream, round(s.end / FRAME_SECONDS)) for s in segments] == [
        (Stream.SYSTEM, 22),
        (Stream.MIC, 52),
    ]


def test_open_segments_on_every_stream_are_flushed_at_the_end() -> None:
    segments = collect_segments(
        {Stream.SYSTEM: [LOUD] * 20, Stream.MIC: [LOUD] * 10}
    )

    assert sorted(s.stream for s in segments) == [Stream.MIC, Stream.SYSTEM]


def test_each_stream_gets_its_own_detector() -> None:
    made: list[LoudnessDetector] = []

    def factory() -> LoudnessDetector:
        made.append(LoudnessDetector())
        return made[-1]

    async def run_all() -> None:
        source = ScriptedSource({Stream.SYSTEM: [QUIET], Stream.MIC: [QUIET]})
        async for _ in segment_stream(source, factory, CONFIG):
            pass

    asyncio.run(run_all())

    assert len(made) == 2


def test_closing_the_segment_stream_closes_its_source_immediately() -> None:
    # Live capture has a child process to stop, so this can't wait for a later tick.
    closed = False

    class ClosableSource:
        async def __aiter__(self) -> AsyncGenerator[Frame, None]:
            nonlocal closed
            try:
                for _ in range(1000):
                    pcm = np.full(FRAME_SAMPLES, LOUD, dtype=np.float32)
                    yield Frame(Stream.MIC, pcm, 0.0)
            finally:
                closed = True

    async def run_all() -> bool:
        stream = segment_stream(ClosableSource(), LoudnessDetector, CONFIG)
        await anext(stream)
        await stream.aclose()
        return closed

    assert asyncio.run(run_all())


def test_speech_after_a_split_inside_a_pause_is_kept() -> None:
    # The cap lands in a pause that never reaches silence_end_ms, so the short
    # speech after it continues the same talk and isn't subject to the minimum.
    levels = [LOUD] * 92 + [QUIET] * 8 + [LOUD] * 2 + [QUIET] * 20

    segments = run(levels, flush=False)

    assert segments[-1].end == 104 * FRAME_SECONDS


def test_a_split_on_the_last_frame_of_the_cap_keeps_later_speech() -> None:
    # A one-frame window can pick the final frame, leaving nothing open.
    config = CONFIG.model_copy(update={"split_window_ms": 32})
    levels = [STRONG] * 99 + [WEAK] + [STRONG] * 4 + [QUIET] * 20

    segments = run(levels, config, flush=False)

    assert segments[-1].end == 106 * FRAME_SECONDS


def test_no_segment_is_only_silence() -> None:
    # After a split in a pause, only silence remains until the Segment closes.
    levels = [LOUD] * 92 + [QUIET] * 40

    assert len(run(levels, flush=False)) == 1


def test_end_is_exact_when_the_cap_is_reached_on_a_voiced_frame() -> None:
    levels = [STRONG] * 85 + [WEAK] * 3 + [STRONG] * 12 + [QUIET] * 20

    segments = run(levels, flush=False)

    assert segments[-1].end == 102 * FRAME_SECONDS


def test_a_new_segment_can_reuse_silence_left_over_from_the_last_one() -> None:
    # First Segment keeps frames up to 12; 12..19 stay available as pre-roll.
    levels = [LOUD] * 10 + [QUIET] * 10 + [LOUD] * 10 + [QUIET] * 20

    _, second = run(levels, flush=False)

    assert second.start == 15 * FRAME_SECONDS


def test_a_single_voiced_frame_after_a_segment_does_not_start_speech() -> None:
    config = CONFIG.model_copy(update={"min_speech_ms": 0})
    levels = [LOUD] * 10 + [QUIET] * 15 + [LOUD] + [QUIET] * 30

    assert len(run(levels, config, flush=False)) == 1


@pytest.mark.parametrize("seed", range(50))
def test_sustained_speech_is_never_lost_and_cuts_never_overlap(seed: int) -> None:
    rng = random.Random(seed)
    levels: list[float] = []
    sustained: list[int] = []
    for _ in range(40):
        pause = rng.randint(0, 25)
        levels += [QUIET] * pause
        talk = rng.randint(1, 60)
        if talk >= 5:
            sustained += range(len(levels), len(levels) + talk)
        levels += [rng.choice([STRONG, WEAK]) for _ in range(talk)]

    segments = run(levels)

    for earlier, later in zip(segments, segments[1:], strict=False):
        assert earlier.end <= later.start + 1e-9  # Float rounding at a shared edge.
    assert all(s.pcm.size <= 100 * FRAME_SAMPLES for s in segments)
    covered = [
        any(s.start <= i * FRAME_SECONDS < s.end for s in segments) for i in sustained
    ]
    assert all(covered)

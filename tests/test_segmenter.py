import numpy as np
from numpy.typing import NDArray

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, Frame, Stream
from rt_speech.config import SegmenterConfig
from rt_speech.segmenter import Segment, Segmenter

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
        assert earlier.end == later.start
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
    assert segments[1].start == segments[0].end


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

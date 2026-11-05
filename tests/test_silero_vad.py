import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE
from rt_speech.silero_vad import SileroVad
from tests.conftest import read_frames

GOLDEN = Path(__file__).parent / "fixtures" / "silero_golden.json"


def deterministic_signal() -> list[NDArray[np.float32]]:
    """Harmonics under a slow on/off envelope, computed without any randomness."""
    t = np.arange(FRAME_SAMPLES * 200) / SAMPLE_RATE
    gate = np.sin(2 * np.pi * 0.7 * t) > 0
    envelope = gate * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    tone = sum(np.sin(2 * np.pi * 140 * k * t) / k for k in range(1, 12))
    signal = (0.2 * envelope * tone).astype(np.float32)
    return list(signal.reshape(-1, FRAME_SAMPLES))


def test_scores_match_the_official_package() -> None:
    # Reference scores come from silero-vad 6.2.3 itself, so this catches any
    # mistake in how the state and the context carry between frames.
    expected = json.loads(GOLDEN.read_text())
    vad = SileroVad()

    scores = [vad.speech_probability(pcm) for pcm in deterministic_signal()]

    assert scores == pytest.approx(expected, abs=1e-3)


def test_a_reused_capture_buffer_gives_the_same_scores() -> None:
    frames = deterministic_signal()
    fresh_vad = SileroVad()
    fresh = [fresh_vad.speech_probability(pcm) for pcm in frames]

    vad = SileroVad()
    buffer = np.zeros(FRAME_SAMPLES, dtype=np.float32)
    reused = []
    for pcm in frames:
        buffer[:] = pcm
        reused.append(vad.speech_probability(buffer))

    assert reused == fresh


@pytest.mark.parametrize(
    "pcm",
    [
        np.zeros(256, dtype=np.float32),
        np.zeros(FRAME_SAMPLES + 1, dtype=np.float32),
        np.zeros(FRAME_SAMPLES, dtype=np.int16),
        np.zeros(FRAME_SAMPLES, dtype=np.float64),
    ],
    ids=["short", "long", "int16", "float64"],
)
def test_frames_outside_the_contract_are_rejected(pcm: NDArray) -> None:
    with pytest.raises(ValueError, match="512"):
        SileroVad().speech_probability(pcm)


def test_russian_speech_scores_high(russian_speech_wav: Path) -> None:
    vad = SileroVad()

    speech = [vad.speech_probability(pcm) for pcm in read_frames(russian_speech_wav)]

    assert max(speech) > 0.9
    assert sum(p > 0.5 for p in speech) / len(speech) > 0.5


def test_speech_scores_high_and_silence_scores_low(speech_wav: Path) -> None:
    vad = SileroVad()

    speech = [vad.speech_probability(pcm) for pcm in read_frames(speech_wav)]
    silence = [
        vad.speech_probability(np.zeros(FRAME_SAMPLES, dtype=np.float32))
        for _ in range(60)
    ]

    assert max(speech) > 0.9
    assert sum(p > 0.5 for p in speech) / len(speech) > 0.5
    assert max(silence[30:]) < 0.1


def test_two_detectors_do_not_share_state(speech_wav: Path) -> None:
    frames = read_frames(speech_wav)
    noise = np.random.default_rng(0).normal(0, 0.2, (len(frames), FRAME_SAMPLES))

    alone_vad = SileroVad()
    alone = [alone_vad.speech_probability(pcm) for pcm in frames]

    vad, other = SileroVad(), SileroVad()
    interleaved = []
    for pcm, junk in zip(frames, noise.astype(np.float32), strict=True):
        other.speech_probability(junk)
        interleaved.append(vad.speech_probability(pcm))

    assert interleaved == alone

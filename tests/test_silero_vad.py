from pathlib import Path

import numpy as np

from rt_speech.audio import FRAME_SAMPLES
from rt_speech.silero_vad import SileroVad
from tests.conftest import read_frames


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

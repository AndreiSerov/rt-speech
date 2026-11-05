import wave
from pathlib import Path

import numpy as np
import pytest

from rt_speech.audio import SAMPLE_RATE
from rt_speech.debug_segment import main


def with_silence(speech: Path, out: Path, seconds: float = 1.5) -> Path:
    """Speech, a pause long enough to split, then the same speech again."""
    with wave.open(str(speech), "rb") as wav:
        data = wav.readframes(wav.getnframes())
    gap = bytes(int(seconds * SAMPLE_RATE) * 2)
    with wave.open(str(out), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(gap + data + gap + data + gap)
    return out


def test_dumps_each_segment_as_its_own_playable_wav(
    speech_wav: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = with_silence(speech_wav, tmp_path / "input.wav")
    out = tmp_path / "segments"

    assert main([str(source), "--out", str(out)]) == 0

    files = sorted(out.glob("*.wav"))
    assert len(files) == 2
    for file in files:
        with wave.open(str(file), "rb") as wav:
            assert wav.getframerate() == SAMPLE_RATE
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            samples = np.frombuffer(wav.readframes(wav.getnframes()), "<i2")
        assert np.abs(samples).max() > 3000
    assert str(files[0].name) in capsys.readouterr().out


def test_rerunning_replaces_earlier_segments_only(
    speech_wav: Path, tmp_path: Path
) -> None:
    out = tmp_path / "segments"
    out.mkdir()
    (out / "segment_0099_old.wav").write_bytes(b"stale")
    (out / "notes.txt").write_text("keep me")

    assert main([str(speech_wav), "--out", str(out)]) == 0

    assert not (out / "segment_0099_old.wav").exists()
    assert (out / "notes.txt").exists()


def test_a_bad_input_file_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "nope.wav"), "--out", str(tmp_path / "s")]) == 1

    assert "not found" in capsys.readouterr().err


def test_a_bad_config_is_a_clean_error(
    speech_wav: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "bad.toml"
    config.write_text("[segmenter]\ntypo = 1\n")

    code = main(
        [str(speech_wav), "--out", str(tmp_path / "s"), "--config", str(config)]
    )

    assert code == 1
    assert "typo" in capsys.readouterr().err

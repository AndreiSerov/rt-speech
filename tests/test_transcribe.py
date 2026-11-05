from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from rt_speech.config import SttConfig
from rt_speech.stt import TranscriptPiece
from rt_speech.stt_engines import SttSetupError
from rt_speech.transcribe import main


class FakeEngine:
    def __init__(self) -> None:
        self.languages: list[str | None] = []

    def transcribe(
        self, pcm: NDArray[np.float32], language: str | None
    ) -> list[TranscriptPiece]:
        self.languages.append(language)
        return [TranscriptPiece(f"phrase{len(self.languages)}")]


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    fake = FakeEngine()
    monkeypatch.setattr("rt_speech.transcribe.make_transcriber", lambda _: fake)
    return fake


def test_a_file_prints_as_the_client_by_default(
    engine: FakeEngine, speech_wav: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(speech_wav)]) == 0

    assert capsys.readouterr().out == "[client] phrase1\n"


def test_stream_label_mic_prints_as_me(
    engine: FakeEngine, speech_wav: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(speech_wav), "--stream-label", "mic"]) == 0

    assert capsys.readouterr().out == "[me] phrase1\n"


def test_a_second_file_plays_on_the_other_stream(
    engine: FakeEngine,
    speech_wav: Path,
    russian_speech_wav: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(speech_wav), "--other", str(russian_speech_wav)])

    assert code == 0
    out = capsys.readouterr().out
    assert "[client]" in out and "[me]" in out


def test_language_flag_replaces_every_configured_language(
    engine: FakeEngine, speech_wav: Path, tmp_path: Path
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[stt]\nlanguage = "ru"\nsystem_language = "ru"\n')

    assert main([str(speech_wav), "--language", "en", "--config", str(config)]) == 0

    assert engine.languages == ["en"]


def test_language_flag_auto_means_detection(
    engine: FakeEngine, speech_wav: Path
) -> None:
    assert main([str(speech_wav), "--language", "auto"]) == 0

    assert engine.languages == [None]


def test_without_the_flag_the_config_language_applies(
    engine: FakeEngine, speech_wav: Path, tmp_path: Path
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[stt]\nlanguage = "en"\n')

    assert main([str(speech_wav), "--config", str(config)]) == 0

    assert engine.languages == ["en"]


def test_a_missing_file_is_a_clean_error(
    engine: FakeEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path / "nope.wav")]) == 1

    assert "not found" in capsys.readouterr().err


def test_a_missing_engine_is_a_clean_error(
    speech_wav: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_: SttConfig) -> None:
        raise SttSetupError("uv sync --extra mlx")

    monkeypatch.setattr("rt_speech.transcribe.make_transcriber", fail)

    assert main([str(speech_wav)]) == 1

    assert "uv sync --extra mlx" in capsys.readouterr().err


def test_the_file_is_checked_before_loading_an_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_load(_: SttConfig) -> None:
        raise AssertionError("loaded a model for a file that can't be read")

    monkeypatch.setattr("rt_speech.transcribe.make_transcriber", must_not_load)

    assert main([str(tmp_path / "nope.wav")]) == 1

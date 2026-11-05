import asyncio
import os
import signal
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from rt_speech.audio import SAMPLE_RATE
from rt_speech.config import SttConfig
from rt_speech.stt import TranscriptPiece
from rt_speech.stt_engines import SttSetupError
from rt_speech.transcribe import InterruptHandler, main


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

    assert capsys.readouterr().out == (
        "[client] phrase1\n"
        '[suggestion] (dummy) Follow up on "phrase1"\n'
    )


def test_suggestions_can_be_turned_off(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[suggester]\nengine = "off"\n')

    assert main([str(speech_wav), "--config", str(config)]) == 0

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


def test_ctrl_c_finishes_what_was_heard_and_exits_cleanly(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with wave.open(str(speech_wav), "rb") as wav:
        speech = wav.readframes(wav.getnframes())
    long_wav = tmp_path / "long.wav"
    with wave.open(str(long_wav), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(speech * 30)
    config = tmp_path / "config.toml"
    config.write_text("[replay]\nrealtime = true\n")
    threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGINT)).start()

    started = time.monotonic()
    code = main([str(long_wav), "--config", str(config)])

    assert code == 0
    assert time.monotonic() - started < 15
    assert capsys.readouterr().out == (
        "[client] phrase1\n"
        '[suggestion] (dummy) Follow up on "phrase1"\n'
    )


def test_a_second_ctrl_c_forces_exit() -> None:
    stop = asyncio.Event()
    forced: list[int] = []
    handler = InterruptHandler(stop, forced.append)

    handler()
    assert stop.is_set()
    assert forced == []

    handler()
    assert forced == [130]


def test_a_failure_while_running_is_a_clean_error(
    engine: FakeEngine,
    speech_wav: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(*_: object, **__: object) -> None:
        raise OSError("device gone")

    monkeypatch.setattr("rt_speech.transcribe.run_pipeline", fail)

    assert main([str(speech_wav)]) == 1

    assert "device gone" in capsys.readouterr().err


def test_a_closed_pipe_exits_quietly(
    engine: FakeEngine,
    speech_wav: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(*_: object, **__: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr("rt_speech.transcribe.run_pipeline", fail)
    monkeypatch.setattr("rt_speech.transcribe._silence_stdout", lambda: None)

    assert main([str(speech_wav)]) == 0

    assert capsys.readouterr().err == ""


def test_ctrl_c_while_the_model_loads_exits_cleanly(
    speech_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(_: SttConfig) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("rt_speech.transcribe.make_transcriber", interrupted)

    assert main([str(speech_wav)]) == 130


def test_a_directory_as_input_is_a_clean_error(
    engine: FakeEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path)]) == 1

    assert str(tmp_path) in capsys.readouterr().err


def test_a_directory_as_config_is_a_clean_error(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([str(speech_wav), "--config", str(tmp_path)]) == 1

    assert str(tmp_path) in capsys.readouterr().err

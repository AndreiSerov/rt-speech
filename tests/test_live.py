import wave
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from rt_speech.audio import SAMPLE_RATE, Frame, Stream
from rt_speech.file_source import FileSource
from rt_speech.live import main
from rt_speech.stt import TranscriptPiece

STARTED = datetime(2026, 11, 6, 14, 30, 5)


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def transcribe(
        self, pcm: NDArray[np.float32], language: str | None
    ) -> list[TranscriptPiece]:
        self.calls += 1
        return [TranscriptPiece(f"phrase{self.calls}")]


class FakeCapture:
    """Stands in for the Swift CLI: replays a WAV as the system Stream."""

    def __init__(self, wav: Path) -> None:
        self.wav = wav
        self.started = False

    def __call__(self, _binary: Path) -> "FakeCapture":
        return self

    async def __aiter__(self) -> AsyncGenerator[Frame, None]:
        self.started = True
        async for frame in FileSource([(self.wav, Stream.SYSTEM)]):
            yield frame


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> FakeEngine:
    fake = FakeEngine()
    monkeypatch.setattr("rt_speech.transcribe.make_transcriber", lambda _: fake)
    return fake


@pytest.fixture
def capture(
    speech_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FakeCapture:
    """Five quiet seconds, then speech."""
    with wave.open(str(speech_wav), "rb") as wav:
        speech = wav.readframes(wav.getnframes())
    path = tmp_path / "call.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(2 * SAMPLE_RATE * 5) + speech)
    fake = FakeCapture(path)
    monkeypatch.setattr("rt_speech.live.MacCaptureSource", fake)
    return fake


@pytest.fixture(autouse=True)
def keep_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("rt_speech.transcribe._silence_stdout", lambda: None)


def _config(tmp_path: Path, extra: str = "") -> tuple[Path, Path]:
    root = tmp_path / "transcripts"
    config = tmp_path / "config.toml"
    config.write_text(f'[transcript]\ndir = "{root}"\n{extra}')
    return config, root


def test_a_call_is_saved_by_the_time_it_started_with_timecodes_from_capture_start(
    engine: FakeEngine, capture: FakeCapture, tmp_path: Path
) -> None:
    config, root = _config(tmp_path)

    assert main(["--config", str(config)], now=lambda: STARTED) == 0

    saved = root / "live" / "2026-11-06_14-30-05.txt"
    assert saved.read_text(encoding="utf-8") == (
        "Встреча 2026-11-06 14:30:05\n\n[00:00:04] [client] phrase1\n"
    )


def test_a_directory_that_cant_be_used_stops_the_call_before_capture_starts(
    engine: FakeEngine,
    capture: FakeCapture,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "transcripts").write_text("a file where the directory should go")
    config, _ = _config(tmp_path)

    assert main(["--config", str(config)]) == 1

    assert capsys.readouterr().err.startswith("error: ")
    assert not capture.started


def test_transcripts_can_be_turned_off(
    engine: FakeEngine, capture: FakeCapture, tmp_path: Path
) -> None:
    config, root = _config(tmp_path, "enabled = false\n")

    assert main(["--config", str(config)], now=lambda: STARTED) == 0

    assert not root.exists()

import asyncio
import os
import signal
import threading
import time
import wave
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from rt_speech.audio import SAMPLE_RATE
from rt_speech.config import SttConfig
from rt_speech.stt import TranscriptPiece
from rt_speech.stt_engines import SttSetupError
from rt_speech.transcribe import InterruptHandler, main
from rt_speech.turns import Speaker, Turn


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


@pytest.fixture(autouse=True)
def keep_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real one replaces the test process's stdout.
    monkeypatch.setattr("rt_speech.transcribe._silence_stdout", lambda: None)


def test_a_file_prints_as_the_client_by_default(
    engine: FakeEngine, speech_wav: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(speech_wav)]) == 0

    assert capsys.readouterr().out == (
        '[client] phrase1\n[suggestion] (dummy) Follow up on "phrase1"\n'
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
        '[client] phrase1\n[suggestion] (dummy) Follow up on "phrase1"\n'
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(*_: object, **__: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr("rt_speech.transcribe.run_pipeline", fail)
    config = tmp_path / "config.toml"
    config.write_text("[transcript]\nenabled = false\n")

    assert main([str(speech_wav), "--config", str(config)]) == 0

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


class SlowSuggester:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    async def suggest(self, turns: Sequence[Turn]) -> list[str]:
        await asyncio.sleep(self.seconds)
        return ["late"]


def _pipeline_that_ends_with(
    failure: Exception | None,
) -> Callable[..., object]:
    """Closes one Client Turn, gives background work a moment, then ends."""

    async def fake(
        _source: object,
        _vad: object,
        _stt: object,
        _config: object,
        on_turn: Callable[[Turn], None],
        **_: object,
    ) -> None:
        on_turn(Turn(Speaker.CLIENT, "phrase1", 0.0, 1.0))
        await asyncio.sleep(0.01)
        if failure is not None:
            raise failure

    return fake


def test_the_last_suggestion_is_waited_for_at_the_end_of_a_run(
    engine: FakeEngine,
    speech_wav: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "rt_speech.transcribe.make_suggester", lambda _: SlowSuggester(0.1)
    )
    monkeypatch.setattr(
        "rt_speech.transcribe.run_pipeline", _pipeline_that_ends_with(None)
    )

    assert main([str(speech_wav)]) == 0

    assert "[suggestion] late\n" in capsys.readouterr().out


def test_a_capture_failure_is_reported_without_waiting_for_a_suggestion(
    engine: FakeEngine,
    speech_wav: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "rt_speech.transcribe.make_suggester", lambda _: SlowSuggester(3.0)
    )
    monkeypatch.setattr(
        "rt_speech.transcribe.run_pipeline",
        _pipeline_that_ends_with(OSError("device gone")),
    )

    started = time.monotonic()
    code = main([str(speech_wav)])

    assert code == 1
    assert time.monotonic() - started < 2
    assert "device gone" in capsys.readouterr().err


def test_a_closed_pipe_for_a_suggestion_does_not_hide_a_capture_failure(
    engine: FakeEngine,
    speech_wav: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def closed_pipe(_: str) -> None:
        raise BrokenPipeError

    monkeypatch.setattr("rt_speech.transcribe._print_suggestion", closed_pipe)
    monkeypatch.setattr(
        "rt_speech.transcribe.run_pipeline",
        _pipeline_that_ends_with(OSError("device gone")),
    )

    assert main([str(speech_wav)]) == 1

    assert "device gone" in capsys.readouterr().err


def test_the_suggester_settings_reach_the_run(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[suggester]\ntimeout_seconds = 0.01\n")
    monkeypatch.setattr(
        "rt_speech.transcribe.make_suggester", lambda _: SlowSuggester(0.5)
    )
    monkeypatch.setattr(
        "rt_speech.transcribe.run_pipeline", _pipeline_that_ends_with(None)
    )

    assert main([str(speech_wav), "--config", str(config)]) == 0

    assert "[suggestion]" not in capsys.readouterr().out


def test_a_failed_run_leaves_nothing_for_the_final_flush_to_trip_over(
    engine: FakeEngine,
    speech_wav: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failed print stays in stdout's buffer. Flushing it at exit into a closed pipe
    # would replace exit code 1 with 120 and add noise after the real error.
    silenced: list[bool] = []
    monkeypatch.setattr(
        "rt_speech.transcribe._silence_stdout", lambda: silenced.append(True)
    )
    monkeypatch.setattr(
        "rt_speech.transcribe.run_pipeline",
        _pipeline_that_ends_with(OSError("device gone")),
    )

    assert main([str(speech_wav)]) == 1

    assert silenced == [True]


STARTED = datetime(2026, 11, 6, 14, 30, 5)


def _transcript_config(tmp_path: Path, extra: str = "") -> tuple[Path, Path]:
    """A config file whose Transcripts go under tmp_path. Returns it and the root."""
    root = tmp_path / "transcripts"
    config = tmp_path / "config.toml"
    config.write_text(f'[transcript]\ndir = "{root}"\n{extra}')
    return config, root


def _closing(*turns: Turn) -> Callable[..., object]:
    """A pipeline that closes the given Turns, in order, and ends."""

    async def fake(
        _source: object,
        _vad: object,
        _stt: object,
        _config: object,
        on_turn: Callable[[Turn], None],
        **_: object,
    ) -> None:
        for turn in turns:
            on_turn(turn)

    return fake


def test_a_run_saves_its_turns_in_a_transcript_named_by_its_start(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = _transcript_config(tmp_path)
    monkeypatch.setattr(
        "rt_speech.transcribe.run_pipeline",
        _closing(
            Turn(Speaker.CLIENT, "Привет", 12.9, 14.0),
            Turn(Speaker.ME, "Hello", 11.0, 12.0),
        ),
    )

    code = main([str(speech_wav), "--config", str(config)], now=lambda: STARTED)

    assert code == 0
    saved = root / "transcribe" / "en_2026-11-06_14-30-05.txt"
    assert saved.read_text(encoding="utf-8") == (
        "Файл en.wav, обработан 2026-11-06 14:30:05\n"
        "\n"
        "[00:00:12] [client] Привет\n"
        "[00:00:11] [me] Hello\n"
    )


def test_other_lists_both_files_and_the_name_uses_the_main_one(
    engine: FakeEngine,
    speech_wav: Path,
    russian_speech_wav: Path,
    tmp_path: Path,
) -> None:
    config, root = _transcript_config(tmp_path)

    code = main(
        [str(speech_wav), "--other", str(russian_speech_wav), "--config", str(config)],
        now=lambda: STARTED,
    )

    assert code == 0
    saved = root / "transcribe" / "en_2026-11-06_14-30-05.txt"
    first_line = saved.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "Файлы en.wav и ru.wav, обработаны 2026-11-06 14:30:05"


def test_hours_keep_counting_past_a_day(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = _transcript_config(tmp_path)
    monkeypatch.setattr(
        "rt_speech.transcribe.run_pipeline",
        _closing(Turn(Speaker.CLIENT, "late", 90061.9, 90065.0)),
    )

    assert main([str(speech_wav), "--config", str(config)], now=lambda: STARTED) == 0

    saved = root / "transcribe" / "en_2026-11-06_14-30-05.txt"
    assert (
        saved.read_text(encoding="utf-8").splitlines()[2] == "[25:01:01] [client] late"
    )


def test_suggestions_stay_out_of_the_transcript(
    engine: FakeEngine, speech_wav: Path, tmp_path: Path
) -> None:
    config, root = _transcript_config(tmp_path)

    assert main([str(speech_wav), "--config", str(config)], now=lambda: STARTED) == 0

    saved = root / "transcribe" / "en_2026-11-06_14-30-05.txt"
    assert saved.read_text(encoding="utf-8").endswith("[client] phrase1\n")


def test_the_path_goes_to_stderr_and_stdout_is_unchanged(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, root = _transcript_config(tmp_path, '[suggester]\nengine = "off"\n')

    assert main([str(speech_wav), "--config", str(config)], now=lambda: STARTED) == 0

    captured = capsys.readouterr()
    assert captured.out == "[client] phrase1\n"
    assert f"{root}/transcribe/en_2026-11-06_14-30-05.txt\n" in captured.err


def test_a_run_without_speech_leaves_no_file(
    engine: FakeEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    silence = tmp_path / "silence.wav"
    with wave.open(str(silence), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(2 * SAMPLE_RATE))
    config, root = _transcript_config(tmp_path)

    assert main([str(silence), "--config", str(config)], now=lambda: STARTED) == 0

    assert list(root.rglob("*.txt")) == []
    assert capsys.readouterr().out == ""


def test_transcripts_can_be_turned_off(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, root = _transcript_config(tmp_path, "enabled = false\n")

    assert main([str(speech_wav), "--config", str(config)], now=lambda: STARTED) == 0

    assert not root.exists()
    assert "Transcript" not in capsys.readouterr().err


def _must_not_load_an_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_load(_: SttConfig) -> None:
        raise AssertionError("loaded a model although the Transcript can't be saved")

    monkeypatch.setattr("rt_speech.transcribe.make_transcriber", must_not_load)


def test_a_directory_that_cant_be_created_stops_the_run_before_the_engine_loads(
    speech_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _must_not_load_an_engine(monkeypatch)
    blocker = tmp_path / "transcripts"
    blocker.write_text("a file where the directory should go")
    config, _ = _transcript_config(tmp_path)

    assert main([str(speech_wav), "--config", str(config)]) == 1

    assert capsys.readouterr().err.startswith("error: ")


def test_a_directory_that_cant_be_written_stops_the_run_before_the_engine_loads(
    speech_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _must_not_load_an_engine(monkeypatch)
    config, root = _transcript_config(tmp_path)
    (root / "transcribe").mkdir(parents=True)
    (root / "transcribe").chmod(0o500)

    try:
        code = main([str(speech_wav), "--config", str(config)])
    finally:
        (root / "transcribe").chmod(0o700)

    assert code == 1
    assert capsys.readouterr().err.startswith("error: ")


@pytest.fixture
def two_phrases_wav(speech_wav: Path, tmp_path: Path) -> Path:
    """Speech, a pause long enough to close the Turn, and the same speech again."""
    with wave.open(str(speech_wav), "rb") as wav:
        speech = wav.readframes(wav.getnframes())
    path = tmp_path / "two.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(speech + bytes(2 * SAMPLE_RATE * 4) + speech)
    return path


def test_each_turn_is_on_disk_before_the_next_step(
    two_phrases_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = _transcript_config(tmp_path)
    seen: list[str] = []

    class ReadingEngine:
        def transcribe(
            self, pcm: NDArray[np.float32], language: str | None
        ) -> list[TranscriptPiece]:
            files = list(root.rglob("*.txt"))
            seen.append(files[0].read_text(encoding="utf-8") if files else "")
            return [TranscriptPiece(f"phrase{len(seen)}")]

    monkeypatch.setattr(
        "rt_speech.transcribe.make_transcriber", lambda _: ReadingEngine()
    )

    assert main([str(two_phrases_wav), "--config", str(config)]) == 0

    assert seen[0] == ""
    assert seen[1].endswith("[00:00:00] [client] phrase1\n")


class ClosedPipe:
    def write(self, _: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        raise BrokenPipeError


def test_a_turn_is_saved_before_it_is_printed(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = _transcript_config(tmp_path)
    monkeypatch.setattr("sys.stdout", ClosedPipe())

    assert main([str(speech_wav), "--config", str(config)], now=lambda: STARTED) == 0

    saved = root / "transcribe" / "en_2026-11-06_14-30-05.txt"
    assert saved.read_text(encoding="utf-8").endswith("[client] phrase1\n")


def test_turns_closed_before_a_failed_run_stay_in_the_file(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, root = _transcript_config(tmp_path)
    closing = _closing(Turn(Speaker.CLIENT, "before", 1.0, 2.0))

    async def closes_then_fails(*args: object, **kwargs: object) -> None:
        await closing(*args, **kwargs)
        raise OSError("device gone")

    monkeypatch.setattr("rt_speech.transcribe.run_pipeline", closes_then_fails)

    code = main([str(speech_wav), "--config", str(config)], now=lambda: STARTED)

    assert code == 1
    assert "device gone" in capsys.readouterr().err
    saved = root / "transcribe" / "en_2026-11-06_14-30-05.txt"
    assert saved.read_text(encoding="utf-8").endswith("[00:00:01] [client] before\n")


def test_a_write_failure_mid_run_is_reported_once_and_turns_keep_printing(
    two_phrases_wav: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, root = _transcript_config(tmp_path, '[suggester]\nengine = "off"\n')
    directory = root / "transcribe"
    calls = 0

    class LockingEngine:
        def transcribe(
            self, pcm: NDArray[np.float32], language: str | None
        ) -> list[TranscriptPiece]:
            nonlocal calls
            calls += 1
            directory.chmod(0o500)
            return [TranscriptPiece(f"phrase{calls}")]

    monkeypatch.setattr(
        "rt_speech.transcribe.make_transcriber", lambda _: LockingEngine()
    )

    try:
        code = main([str(two_phrases_wav), "--config", str(config)])
    finally:
        directory.chmod(0o700)

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == "[client] phrase1\n[client] phrase2\n"
    assert captured.err.count("can't save the Transcript") == 1
    assert list(directory.iterdir()) == []


def test_an_existing_transcript_is_never_overwritten(
    engine: FakeEngine,
    speech_wav: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, root = _transcript_config(tmp_path)
    taken = root / "transcribe" / "en_2026-11-06_14-30-05.txt"
    taken.parent.mkdir(parents=True)
    taken.write_text("an earlier meeting\n", encoding="utf-8")

    code = main([str(speech_wav), "--config", str(config)], now=lambda: STARTED)

    captured = capsys.readouterr()
    assert code == 0
    assert taken.read_text(encoding="utf-8") == "an earlier meeting\n"
    assert "can't save the Transcript" in captured.err
    assert "[client] phrase1" in captured.out

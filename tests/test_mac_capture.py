import asyncio
import logging
import os
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, Frame, Stream
from rt_speech.config import Config
from rt_speech.mac_capture import CaptureError, MacCaptureSource
from rt_speech.pipeline import run_pipeline
from rt_speech.stt import TranscriptPiece
from rt_speech.turns import Turn

REPO_ROOT = Path(__file__).parent.parent
TIMEOUT = 10.0
RECORD_SIZE = 10 + FRAME_SAMPLES * 4


def fake_binary(tmp_path: Path, body: str) -> Path:
    """A script standing in for rt-capture, with `record()` available to it."""
    path = tmp_path / "fake-capture"
    path.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """\
            import os, signal, struct, sys, time
            def record(stream=0, ts=0.0, value=0.25, version=1):
                sys.stdout.buffer.write(struct.pack("<BBd", version, stream, ts))
                sys.stdout.buffer.write(struct.pack("<f", value) * 512)
                sys.stdout.buffer.flush()
            """
        )
        + textwrap.dedent(body)
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def collect(source: MacCaptureSource) -> list[Frame]:
    async def run() -> list[Frame]:
        return [frame async for frame in source]

    return asyncio.run(asyncio.wait_for(run(), TIMEOUT))


def test_records_become_frames(tmp_path: Path) -> None:
    binary = fake_binary(
        tmp_path, "record(stream=0, ts=1.5, value=0.25)\nrecord(stream=1, ts=2.5, value=-0.5)\n"
    )

    first, second = collect(MacCaptureSource(binary))

    assert first.stream is Stream.SYSTEM
    assert first.timestamp == 1.5
    assert first.pcm.shape == (FRAME_SAMPLES,)
    assert first.pcm.dtype == np.float32
    assert np.all(first.pcm == 0.25)
    assert second.stream is Stream.MIC
    assert second.timestamp == 2.5
    assert np.all(second.pcm == -0.5)


def test_a_clean_exit_just_ends_the_source(tmp_path: Path) -> None:
    assert collect(MacCaptureSource(fake_binary(tmp_path, "pass\n"))) == []


def test_a_missing_binary_says_how_to_build_it(tmp_path: Path) -> None:
    with pytest.raises(CaptureError, match="swift build"):
        collect(MacCaptureSource(tmp_path / "nope"))


def test_a_failed_exit_reports_the_processes_own_message(tmp_path: Path) -> None:
    binary = fake_binary(
        tmp_path,
        """
        record()
        print("Grant Screen Recording in System Settings", file=sys.stderr, flush=True)
        sys.exit(2)
        """,
    )

    with pytest.raises(CaptureError, match="Grant Screen Recording"):
        collect(MacCaptureSource(binary))


def test_a_cut_off_record_is_an_error(tmp_path: Path) -> None:
    binary = fake_binary(
        tmp_path,
        """
        record()
        sys.stdout.buffer.write(b"partial")
        sys.stdout.buffer.flush()
        """,
    )

    with pytest.raises(CaptureError, match="cut off"):
        collect(MacCaptureSource(binary))


def test_an_unknown_format_version_is_an_error(tmp_path: Path) -> None:
    binary = fake_binary(tmp_path, "record(version=9)\n")

    with pytest.raises(CaptureError, match="version 9"):
        collect(MacCaptureSource(binary))


def test_an_unknown_stream_is_an_error(tmp_path: Path) -> None:
    binary = fake_binary(tmp_path, "record(stream=7)\n")

    with pytest.raises(CaptureError, match="stream 7"):
        collect(MacCaptureSource(binary))


def test_its_stderr_goes_to_the_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    binary = fake_binary(tmp_path, 'print("device changed", file=sys.stderr, flush=True)\n')

    with caplog.at_level(logging.INFO):
        collect(MacCaptureSource(binary))

    assert "device changed" in caplog.text


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _run_until_first_frame_then_close(binary: Path, pid_file: Path) -> int:
    async def run() -> None:
        source = aiter(MacCaptureSource(binary))
        await anext(source)
        await source.aclose()

    asyncio.run(asyncio.wait_for(run(), TIMEOUT))
    return int(pid_file.read_text())


def test_closing_stops_the_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    binary = fake_binary(
        tmp_path,
        f"""
        open({str(pid_file)!r}, "w").write(str(os.getpid()))
        record()
        time.sleep(60)
        """,
    )

    pid = _run_until_first_frame_then_close(binary, pid_file)

    assert not _alive(pid)


def test_a_process_ignoring_termination_is_killed(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    binary = fake_binary(
        tmp_path,
        f"""
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        open({str(pid_file)!r}, "w").write(str(os.getpid()))
        record()
        time.sleep(60)
        """,
    )

    pid = _run_until_first_frame_then_close(binary, pid_file)

    assert not _alive(pid)


def test_ctrl_c_in_the_terminal_does_not_reach_the_process(tmp_path: Path) -> None:
    # The process must keep producing while the pipeline finishes up, so it needs its
    # own session: a terminal sends Ctrl-C to the whole foreground process group.
    binary = fake_binary(
        tmp_path,
        """
        value = 1.0 if os.getsid(0) == os.getpid() else -1.0
        record(value=value)
        """,
    )

    [frame] = collect(MacCaptureSource(binary))

    assert np.all(frame.pcm == 1.0)


# The real binary, in its permission-free test mode.


@pytest.fixture(scope="session")
def real_binary() -> Path:
    package = REPO_ROOT / "native" / "capture"
    subprocess.run(
        ["swift", "build", "-c", "release", "--package-path", str(package)],
        check=True,
        capture_output=True,
    )
    return package / ".build" / "release" / "rt-capture"


def _dominant_hertz(frames: list[Frame]) -> float:
    samples = np.concatenate([f.pcm for f in frames])
    spectrum = np.abs(np.fft.rfft(samples))
    return float(np.fft.rfftfreq(samples.size, 1 / SAMPLE_RATE)[int(np.argmax(spectrum))])


def test_the_real_binary_produces_both_streams_over_the_wire(real_binary: Path) -> None:
    frames = collect(MacCaptureSource(real_binary, "--test-signal", "--duration", "1"))

    system = [f for f in frames if f.stream is Stream.SYSTEM]
    mic = [f for f in frames if f.stream is Stream.MIC]
    assert 28 <= len(system) <= 33
    assert 28 <= len(mic) <= 33
    assert _dominant_hertz(system) == pytest.approx(440, abs=5)
    assert _dominant_hertz(mic) == pytest.approx(880, abs=5)
    for stream_frames in (system, mic):
        gaps = np.diff([f.timestamp for f in stream_frames])
        assert gaps == pytest.approx(FRAME_SAMPLES / SAMPLE_RATE, abs=1e-6)


def test_the_real_binary_paces_itself_like_live_audio(real_binary: Path) -> None:
    started = time.monotonic()
    collect(MacCaptureSource(real_binary, "--test-signal", "--duration", "1"))

    assert 0.8 < time.monotonic() - started < 3


def test_the_real_binary_exits_when_nobody_reads_its_output(real_binary: Path) -> None:
    # If the parent crashes, capture must not keep running unattended.
    process = subprocess.Popen(
        [str(real_binary), "--test-signal", "--duration", "60"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    assert len(process.stdout.read(RECORD_SIZE)) == RECORD_SIZE

    process.stdout.close()

    process.wait(timeout=5)  # Raises if it keeps running.


def test_the_real_binary_rejects_unknown_flags(real_binary: Path) -> None:
    with pytest.raises(CaptureError, match="--nonsense"):
        collect(MacCaptureSource(real_binary, "--nonsense"))


@pytest.mark.manual
def test_real_devices_produce_both_streams(real_binary: Path) -> None:
    """Needs Screen Recording and Microphone permission for the terminal app."""

    async def run() -> list[Frame]:
        frames: list[Frame] = []
        async for frame in MacCaptureSource(real_binary):
            frames.append(frame)
            if len(frames) >= 200:
                break
        return frames

    frames = asyncio.run(asyncio.wait_for(run(), 30))

    assert {f.stream for f in frames} == {Stream.SYSTEM, Stream.MIC}


class _LoudnessDetector:
    def speech_probability(self, pcm: NDArray[np.float32]) -> float:
        return 1.0 if float(np.abs(pcm).max()) > 0.1 else 0.0


class _FixedTranscriber:
    def transcribe(self, pcm: NDArray[np.float32], language: str | None) -> list[TranscriptPiece]:
        return [TranscriptPiece("heard")]


def _run_live(binary: Path, stop_after_turn: bool = False) -> tuple[list[Turn], BaseException | None]:
    turns: list[Turn] = []

    async def run() -> None:
        stop = asyncio.Event()

        def on_turn(turn: Turn) -> None:
            turns.append(turn)
            if stop_after_turn:
                stop.set()

        await run_pipeline(
            MacCaptureSource(binary),
            _LoudnessDetector,
            _FixedTranscriber(),
            Config(),
            on_turn,
            stop=stop,
        )

    try:
        asyncio.run(asyncio.wait_for(run(), TIMEOUT))
    except CaptureError as e:
        return turns, e
    return turns, None


SPEECH_THEN_SILENCE = """\
    ts = 0.0
    def emit(value, seconds):
        global ts
        for _ in range(int(seconds / 0.032)):
            record(stream=0, ts=ts, value=value)
            ts += 0.032
    emit(0.5, 1.0)
    emit(0.0, 4.0)
    """


def test_stopping_the_pipeline_ends_the_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    binary = fake_binary(
        tmp_path,
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        + textwrap.dedent(SPEECH_THEN_SILENCE)
        + "while True:\n    emit(0.0, 0.032)\n    time.sleep(0.032)\n",
    )

    turns, failure = _run_live(binary, stop_after_turn=True)

    assert [turn.text for turn in turns] == ["heard"]
    assert failure is None
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_a_capture_failure_keeps_what_was_heard_then_reports_itself(tmp_path: Path) -> None:
    binary = fake_binary(
        tmp_path,
        textwrap.dedent(SPEECH_THEN_SILENCE) + "print('device gone', file=sys.stderr)\nsys.exit(1)\n",
    )

    turns, failure = _run_live(binary)

    assert [turn.text for turn in turns] == ["heard"]
    assert isinstance(failure, CaptureError)
    assert "device gone" in str(failure)

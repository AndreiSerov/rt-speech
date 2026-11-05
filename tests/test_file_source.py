import asyncio
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, AudioSource, Frame, Stream
from rt_speech.file_source import FileSource, FileSourceError

FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE


def write_wav(
    path: Path,
    samples: list[int] | np.ndarray,
    *,
    rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return path


def collect(source: AudioSource) -> list[Frame]:
    async def run() -> list[Frame]:
        return [frame async for frame in source]

    return asyncio.run(run())


def test_frames_have_the_contract_shape_and_type(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", np.zeros(FRAME_SAMPLES * 2))

    frames = collect(FileSource([(path, Stream.SYSTEM)]))

    assert len(frames) == 2
    for frame in frames:
        assert frame.stream is Stream.SYSTEM
        assert frame.pcm.shape == (FRAME_SAMPLES,)
        assert frame.pcm.dtype == np.float32


def test_samples_scale_to_unit_range(tmp_path: Path) -> None:
    samples = [-32768, 0, 16384, 32767] + [0] * (FRAME_SAMPLES - 4)
    path = write_wav(tmp_path / "a.wav", samples)

    [frame] = collect(FileSource([(path, Stream.MIC)]))

    assert frame.pcm[:3].tolist() == [-1.0, 0.0, 0.5]
    assert frame.pcm[3] == pytest.approx(1.0, abs=1e-4)


def test_timestamps_are_seconds_from_the_start(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", np.zeros(FRAME_SAMPLES * 3))

    frames = collect(FileSource([(path, Stream.MIC)]))

    assert [f.timestamp for f in frames] == [0.0, FRAME_SECONDS, 2 * FRAME_SECONDS]


def test_last_partial_frame_is_padded_with_silence(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", [16384] * (FRAME_SAMPLES + 10))

    frames = collect(FileSource([(path, Stream.MIC)]))

    assert len(frames) == 2
    assert frames[1].pcm.shape == (FRAME_SAMPLES,)
    assert np.all(frames[1].pcm[:10] == 0.5)
    assert np.all(frames[1].pcm[10:] == 0.0)


def test_empty_file_yields_no_frames(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", [])

    assert collect(FileSource([(path, Stream.MIC)])) == []


def test_a_truncated_file_replays_what_is_there(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", np.zeros(FRAME_SAMPLES * 2))
    path.write_bytes(path.read_bytes()[:-1])

    frames = collect(FileSource([(path, Stream.MIC)]))

    assert len(frames) == 2


def test_two_files_merge_by_timestamp_and_the_longer_one_continues(
    tmp_path: Path,
) -> None:
    system = write_wav(tmp_path / "system.wav", np.zeros(FRAME_SAMPLES * 3))
    mic = write_wav(tmp_path / "mic.wav", np.zeros(FRAME_SAMPLES))

    frames = collect(FileSource([(system, Stream.SYSTEM), (mic, Stream.MIC)]))

    assert [(f.stream, f.timestamp) for f in frames] == [
        (Stream.SYSTEM, 0.0),
        (Stream.MIC, 0.0),
        (Stream.SYSTEM, FRAME_SECONDS),
        (Stream.SYSTEM, 2 * FRAME_SECONDS),
    ]


def test_iterating_again_replays_from_the_start(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", np.zeros(FRAME_SAMPLES))
    source = FileSource([(path, Stream.MIC)])

    assert len(collect(source)) == 1
    assert len(collect(source)) == 1


def test_stopping_early_is_safe(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", np.zeros(FRAME_SAMPLES * 5))

    async def run() -> Frame:
        async for frame in FileSource([(path, Stream.MIC)]):
            return frame
        raise AssertionError("expected a frame")

    assert asyncio.run(run()).timestamp == 0.0


def test_realtime_replay_is_paced_by_timestamps(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", np.zeros(FRAME_SAMPLES * 6))

    started = time.monotonic()
    collect(FileSource([(path, Stream.MIC)], realtime=True))
    elapsed = time.monotonic() - started

    # The last frame starts at 5 frame lengths.
    assert elapsed >= 5 * FRAME_SECONDS * 0.95


def test_default_replay_is_faster_than_realtime(tmp_path: Path) -> None:
    one_minute = SAMPLE_RATE * 60
    path = write_wav(tmp_path / "a.wav", np.zeros(one_minute))

    started = time.monotonic()
    collect(FileSource([(path, Stream.MIC)]))

    assert time.monotonic() - started < 10


def test_fast_replay_lets_other_tasks_run(tmp_path: Path) -> None:
    path = write_wav(tmp_path / "a.wav", np.zeros(FRAME_SAMPLES * 3))

    async def run() -> list[str]:
        events: list[str] = []

        async def other() -> None:
            events.append("other")

        task = asyncio.create_task(other())
        async for _ in FileSource([(path, Stream.MIC)]):
            events.append("frame")
        await task
        return events

    assert asyncio.run(run())[0] == "other"


@pytest.mark.parametrize(
    ("rate", "channels", "expected"),
    [(44_100, 1, "44100 Hz"), (SAMPLE_RATE, 2, "2 channel")],
)
def test_wrong_format_is_rejected_with_a_conversion_hint(
    tmp_path: Path, rate: int, channels: int, expected: str
) -> None:
    path = write_wav(tmp_path / "a.wav", np.zeros(1024), rate=rate, channels=channels)

    with pytest.raises(FileSourceError, match=expected) as error:
        FileSource([(path, Stream.MIC)])

    assert "ffmpeg" in str(error.value)


def test_wrong_sample_width_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(bytes(1024))

    with pytest.raises(FileSourceError, match="8-bit"):
        FileSource([(path, Stream.MIC)])


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileSourceError, match="not found"):
        FileSource([(tmp_path / "nope.wav", Stream.MIC)])


@pytest.mark.parametrize("content", [b"", b"not a wav file at all"])
def test_non_wav_file_is_rejected(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "a.wav"
    path.write_bytes(content)

    with pytest.raises(FileSourceError, match="not a readable PCM WAV"):
        FileSource([(path, Stream.MIC)])


def test_no_inputs_are_rejected() -> None:
    with pytest.raises(FileSourceError, match="at least one"):
        FileSource([])


def test_two_files_for_one_stream_are_rejected(tmp_path: Path) -> None:
    a = write_wav(tmp_path / "a.wav", np.zeros(FRAME_SAMPLES))
    b = write_wav(tmp_path / "b.wav", np.zeros(FRAME_SAMPLES))

    with pytest.raises(FileSourceError, match="only one file"):
        FileSource([(a, Stream.MIC), (b, Stream.MIC)])

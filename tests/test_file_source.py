import asyncio
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE, AudioSource, Frame, Stream
from rt_speech.file_source import FileSource, FileSourceError
from tests.conftest import tone, write_mp3

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


def dominant_hz(frames: list[Frame]) -> float:
    pcm = np.concatenate([frame.pcm for frame in frames])
    spectrum = np.abs(np.fft.rfft(pcm))
    return float(np.argmax(spectrum) * SAMPLE_RATE / pcm.size)


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


def test_an_mp3_replays_as_16_khz_mono_frames(tmp_path: Path) -> None:
    path = write_mp3(tmp_path / "a.mp3", tone(1.0, 44_100), rate=44_100, channels=2)

    frames = collect(FileSource([(path, Stream.SYSTEM)]))

    assert 30 <= len(frames) <= 36
    for frame in frames:
        assert frame.stream is Stream.SYSTEM
        assert frame.pcm.shape == (FRAME_SAMPLES,)
        assert frame.pcm.dtype == np.float32
    assert [f.timestamp for f in frames[:3]] == [0.0, FRAME_SECONDS, 2 * FRAME_SECONDS]
    assert dominant_hz(frames) == pytest.approx(440, abs=10)


@pytest.mark.parametrize("rate", [8_000, 16_000, 22_050, 44_100, 48_000])
@pytest.mark.parametrize("channels", [1, 2])
def test_an_mp3_of_any_rate_and_channel_count_keeps_its_pitch_and_length(
    tmp_path: Path, rate: int, channels: int
) -> None:
    path = write_mp3(tmp_path / "a.mp3", tone(2.0, rate), rate=rate, channels=channels)

    frames = collect(FileSource([(path, Stream.MIC)]))

    assert 61 <= len(frames) <= 68  # 2 s is 62.5 frames, plus the encoder's padding
    assert dominant_hz(frames) == pytest.approx(440, abs=10)


def test_the_tail_of_an_mp3_is_not_lost(tmp_path: Path) -> None:
    quiet_then_loud = np.concatenate([np.zeros(22_050), tone(0.5, 44_100)])
    path = write_mp3(tmp_path / "a.mp3", quiet_then_loud)

    frames = collect(FileSource([(path, Stream.SYSTEM)]))

    loud = [i for i, frame in enumerate(frames) if np.abs(frame.pcm).max() > 0.25]
    # Half a second of tone is about 15 frames, and it ends at the end of the file.
    assert 14 <= len(loud) <= 17
    assert loud[-1] >= len(frames) - 3


def test_the_mp3_extension_is_matched_in_any_case(tmp_path: Path) -> None:
    path = write_mp3(tmp_path / "LOUD.MP3", tone(0.5, 44_100))

    assert len(collect(FileSource([(path, Stream.SYSTEM)]))) > 10


def test_an_mp3_and_a_wav_can_play_on_different_streams(tmp_path: Path) -> None:
    mp3 = write_mp3(tmp_path / "a.mp3", tone(0.5, 44_100))
    wav = write_wav(tmp_path / "b.wav", np.zeros(FRAME_SAMPLES * 3))

    frames = collect(FileSource([(mp3, Stream.SYSTEM), (wav, Stream.MIC)]))

    assert {f.stream for f in frames} == {Stream.SYSTEM, Stream.MIC}
    assert [f.timestamp for f in frames] == sorted(f.timestamp for f in frames)


def test_stopping_an_mp3_early_is_safe(tmp_path: Path) -> None:
    path = write_mp3(tmp_path / "a.mp3", tone(2.0, 44_100))

    async def first_frame() -> Frame:
        async for frame in FileSource([(path, Stream.SYSTEM)]):
            return frame
        raise AssertionError("no frames")

    assert asyncio.run(first_frame()).timestamp == 0.0


def test_a_missing_mp3_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileSourceError, match="not found"):
        FileSource([(tmp_path / "nope.mp3", Stream.SYSTEM)])


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"this is not audio at all, just text " * 100, id="text"),
        pytest.param(b"ID3\x04\x00\x00\x00\x00\x00\x00", id="tag-only"),
    ],
)
def test_an_mp3_without_audio_is_rejected_on_construction(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "a.mp3"
    path.write_bytes(content)

    with pytest.raises(FileSourceError, match=r"a\.mp3"):
        FileSource([(path, Stream.SYSTEM)])


def test_damage_partway_through_an_mp3_replays_what_came_before_then_fails(
    tmp_path: Path,
) -> None:
    good = write_mp3(tmp_path / "good.mp3", tone(2.0, 44_100)).read_bytes()
    path = tmp_path / "a.mp3"
    path.write_bytes(good[:20_000] + bytes(4000) + good[24_000:])
    source = FileSource([(path, Stream.SYSTEM)])
    heard: list[Frame] = []

    async def replay() -> None:
        async for frame in source:
            heard.append(frame)

    with pytest.raises(FileSourceError, match=r"a\.mp3"):
        asyncio.run(replay())

    assert len(heard) > 10

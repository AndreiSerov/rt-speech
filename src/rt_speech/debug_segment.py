"""Dumps each Segment of a WAV or MP3 file as its own WAV, to judge the cuts by ear.

Usage: python -m rt_speech.debug_segment input.wav --out segments/
"""

import argparse
import asyncio
import re
import sys
import tempfile
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rt_speech.audio import SAMPLE_RATE, Stream
from rt_speech.config import Config, ConfigError, load_config
from rt_speech.file_source import FileSource, FileSourceError
from rt_speech.logging_setup import setup_logging
from rt_speech.segmenter import Segment, segment_stream
from rt_speech.silero_vad import SileroVad

_CUT_NAME = re.compile(r"segment_\d{4,}_\d+\.\d{3}s-\d+\.\d{3}s\.wav")


def _write_wav(path: Path, segment: Segment) -> None:
    samples = np.clip(np.round(segment.pcm * 32768), -32768, 32767)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(samples.astype("<i2").tobytes())


async def _dump(source: FileSource, config: Config, out: Path) -> int:
    count = 0
    async for segment in segment_stream(source, SileroVad, config.segmenter):
        count += 1
        name = f"segment_{count:04d}_{segment.start:.3f}s-{segment.end:.3f}s.wav"
        _write_wav(out / name, segment)
        print(f"{name}  ({segment.end - segment.start:.2f} s)")
    return count


def _is_earlier_cut(path: Path) -> bool:
    return _CUT_NAME.fullmatch(path.name) is not None


def _run(input_path: Path, out: Path, config: Config) -> int:
    source = FileSource([(input_path, Stream.SYSTEM)], realtime=config.replay.realtime)
    if _is_earlier_cut(input_path) and out.exists() and input_path.parent.samefile(out):
        raise FileSourceError(
            f"{input_path} would be replaced by this run. Choose a different --out."
        )
    out.mkdir(parents=True, exist_ok=True)
    # Cuts are staged, so a failed run can't lose the previous ones.
    with tempfile.TemporaryDirectory(dir=out, prefix=".staging-") as staging:
        staged = Path(staging)
        count = asyncio.run(_dump(source, config, staged))
        for old in out.iterdir():
            if old.is_file() and _is_earlier_cut(old):
                old.unlink()
        for new in staged.iterdir():
            new.replace(out / new.name)
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="16 kHz mono 16-bit WAV file, or an MP3 of any format"
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        setup_logging(config.logging.level)
        count = _run(args.input, args.out, config)
    except (ConfigError, FileSourceError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"{count} segment(s) in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

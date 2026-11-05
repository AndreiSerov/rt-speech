"""Dumps each Segment of a WAV file as its own WAV, to judge the cuts by ear.

Usage: python -m rt_speech.debug_segment input.wav --out segments/
"""

import argparse
import asyncio
import sys
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


def _write_wav(path: Path, segment: Segment) -> None:
    samples = np.clip(segment.pcm, -1.0, 1.0) * 32767
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="16 kHz mono 16-bit WAV file")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        setup_logging(config.logging.level)
        source = FileSource(
            [(args.input, Stream.SYSTEM)], realtime=config.replay.realtime
        )
        args.out.mkdir(parents=True, exist_ok=True)
    except (ConfigError, FileSourceError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    # Earlier cuts would otherwise mix with these when comparing thresholds.
    for old in args.out.glob("segment_*.wav"):
        old.unlink()

    count = asyncio.run(_dump(source, config, args.out))
    print(f"{count} segment(s) in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

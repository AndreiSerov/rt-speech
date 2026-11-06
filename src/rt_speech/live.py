"""Prints speaker-tagged Turns and Suggestions from a live call: system audio and the microphone.

Usage: python -m rt_speech.live
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from rt_speech.config import ConfigError, load_config
from rt_speech.logging_setup import setup_logging
from rt_speech.mac_capture import MacCaptureSource
from rt_speech.transcribe import run_and_print, with_language
from rt_speech.transcript import TranscriptError, open_transcript


def main(
    argv: Sequence[str] | None = None, *, now: Callable[[], datetime] = datetime.now
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=["ru", "en", "auto"],
        help="overrides every language in the config",
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args(argv)

    try:
        config = with_language(load_config(args.config), args.language)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    setup_logging(config.logging.level)
    try:
        transcript = open_transcript(
            config.transcript,
            subdir="live",
            name_prefix="",
            header=lambda when: f"Встреча {when}",
            started=now(),
        )
    except TranscriptError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    source = MacCaptureSource(config.capture.binary)
    return run_and_print(source, config, transcript)


if __name__ == "__main__":
    sys.exit(main())

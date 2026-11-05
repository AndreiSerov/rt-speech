"""Process-wide logging setup."""

import logging
import sys


def setup_logging(level: str) -> None:
    """Send logs to stderr, keeping stdout free for speaker-tagged turns."""
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        force=True,
    )

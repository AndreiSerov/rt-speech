"""Prints speaker-tagged Turns and Suggestions from WAV files.

Usage: python -m rt_speech.transcribe input.wav --stream-label system
"""

import argparse
import asyncio
import os
import signal
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rt_speech.audio import AudioSource, Stream
from rt_speech.config import Config, ConfigError, SttConfig, load_config
from rt_speech.file_source import FileSource, FileSourceError
from rt_speech.logging_setup import setup_logging
from rt_speech.pipeline import run_pipeline
from rt_speech.silero_vad import SileroVad
from rt_speech.stt import Transcriber
from rt_speech.stt_engines import SttSetupError, make_transcriber
from rt_speech.suggester import SuggestionTrigger, make_suggester
from rt_speech.turns import Turn

_INTERRUPTED = 130


def _silence_stdout() -> None:
    os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


def _print_turn(turn: Turn) -> None:
    print(f"[{turn.speaker}] {turn.text}", flush=True)


def _print_suggestion(suggestion: str) -> None:
    print(f"[suggestion] {suggestion}", flush=True)


def with_language(config: Config, language: str | None) -> Config:
    if language is None:
        return config
    stt: SttConfig = config.stt.model_copy(
        update={"language": language, "system_language": None, "mic_language": None}
    )
    return config.model_copy(update={"stt": stt})


class InterruptHandler:
    """The first Ctrl-C finishes what was heard. The second quits at once, because a
    stuck engine's worker thread would otherwise keep the process alive."""

    def __init__(self, stop: asyncio.Event, force_exit: Callable[[int], object]) -> None:
        self._stop = stop
        self._force_exit = force_exit

    def __call__(self) -> None:
        if self._stop.is_set():
            self._force_exit(_INTERRUPTED)
            return
        self._stop.set()
        print("Finishing what was heard. Ctrl-C again to quit now.", file=sys.stderr)


async def _run(source: AudioSource, transcriber: Transcriber, config: Config) -> None:
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGINT, InterruptHandler(stop, os._exit)
    )
    suggestions = SuggestionTrigger(
        make_suggester(config.suggester),
        _print_suggestion,
        config.suggester,
    )

    def on_turn(turn: Turn) -> None:
        _print_turn(turn)
        suggestions.on_turn(turn)

    try:
        await run_pipeline(source, SileroVad, transcriber, config, on_turn, stop=stop)
    except BaseException:
        # A failure is reported at once, and nothing may replace it.
        suggestions.cancel()
        raise
    await suggestions.finish()


def run_and_print(source: AudioSource, config: Config) -> int:
    """Loads the engine and prints Turns from `source`. Returns the exit code."""
    try:
        transcriber = make_transcriber(config.stt)
    except SttSetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return _INTERRUPTED

    try:
        asyncio.run(_run(source, transcriber, config))
    except BrokenPipeError:
        # The reader of our output left, such as `| head`. Nothing more to say, and
        # the interpreter's final flush of stdout mustn't complain either.
        _silence_stdout()
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="16 kHz mono 16-bit WAV file")
    parser.add_argument(
        "--stream-label",
        choices=[stream.value for stream in Stream],
        default=Stream.SYSTEM.value,
        help="which Stream the input plays on (default: system, the Client)",
    )
    parser.add_argument(
        "--other", type=Path, help="a second WAV file, played on the other Stream"
    )
    parser.add_argument(
        "--language",
        choices=["ru", "en", "auto"],
        help="overrides every language in the config",
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    args = parser.parse_args(argv)

    stream = Stream(args.stream_label)
    inputs = [(args.input, stream)]
    if args.other is not None:
        other = Stream.MIC if stream is Stream.SYSTEM else Stream.SYSTEM
        inputs.append((args.other, other))

    try:
        config = with_language(load_config(args.config), args.language)
        setup_logging(config.logging.level)
        source = FileSource(inputs, realtime=config.replay.realtime)
    except (ConfigError, FileSourceError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return run_and_print(source, config)


if __name__ == "__main__":
    sys.exit(main())

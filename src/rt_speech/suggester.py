"""Suggests what Me could say next, each time a Client Turn closes."""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Protocol

from rt_speech.config import SuggesterConfig
from rt_speech.turns import Speaker, Turn

logger = logging.getLogger(__name__)

_DUMMY_QUOTE_WORDS = 6


class Suggester(Protocol):
    async def suggest(self, turns: Sequence[Turn]) -> list[str]:
        """Returns short Suggestions from recent Turns, oldest first. The last one is
        the Client Turn that just closed. It must not block the event loop, or Turns
        stop too. May raise; the failure is logged and the call goes on."""
        ...


class DummySuggester:
    """Stands in for a real model, and says so. It makes no network calls."""

    async def suggest(self, turns: Sequence[Turn]) -> list[str]:
        words = turns[-1].text.split()
        quote = " ".join(words[-_DUMMY_QUOTE_WORDS:])
        if len(words) > _DUMMY_QUOTE_WORDS:
            quote = f"...{quote}"
        return [f'(dummy) Follow up on "{quote}"']


class _NoSuggester:
    async def suggest(self, turns: Sequence[Turn]) -> list[str]:
        return []


def make_suggester(config: SuggesterConfig) -> Suggester:
    return DummySuggester() if config.engine == "dummy" else _NoSuggester()


class SuggestionTrigger:
    """Asks for Suggestions in the background, so a slow Suggester never holds up
    Turns. Only the latest Client Turn is answered."""

    def __init__(
        self,
        suggester: Suggester,
        on_suggestion: Callable[[str], None],
        *,
        recent_turns: int,
        timeout_seconds: float,
    ) -> None:
        self._suggester = suggester
        self._on_suggestion = on_suggestion
        self._timeout = timeout_seconds
        self._recent: deque[Turn] = deque(maxlen=recent_turns)
        self._running: asyncio.Task[None] | None = None
        self._output_failure: Exception | None = None

    def on_turn(self, turn: Turn) -> None:
        self._raise_output_failure()
        self._recent.append(turn)
        if turn.speaker is not Speaker.CLIENT:
            return
        if self._running is not None:
            self._running.cancel()
        self._running = asyncio.create_task(self._suggest(tuple(self._recent)))

    async def finish(self) -> None:
        """Waits for the Suggestion in progress, at most until its timeout."""
        if self._running is not None:
            await asyncio.wait([self._running])
        self._raise_output_failure()

    def _raise_output_failure(self) -> None:
        if self._output_failure is not None:
            raise self._output_failure

    async def _suggest(self, turns: Sequence[Turn]) -> None:
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._timeout):
                suggestions = await self._suggester.suggest(turns)
        except TimeoutError:
            logger.warning("Suggester timed out after %.1f s", self._timeout)
            return
        except Exception:
            logger.exception("Suggester failed")
            return
        logger.info(
            "Got %d Suggestions in %.2f s", len(suggestions), time.monotonic() - started
        )
        try:
            for suggestion in suggestions:
                self._on_suggestion(suggestion)
        except Exception as e:
            # Nothing awaits this task, so the next call to on_turn or finish raises it.
            self._output_failure = e

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
        stop too, and must stop when cancelled, or finishing a run waits on it. Work
        moved to a thread can't be cancelled, so it needs its own limit. May raise; the
        failure is logged and the call goes on."""
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
    return {"dummy": DummySuggester, "off": _NoSuggester}[config.engine]()


class SuggestionTrigger:
    """Asks for Suggestions in the background, so a slow Suggester never holds up
    Turns. Only the latest Client Turn is answered."""

    def __init__(
        self,
        suggester: Suggester,
        on_suggestion: Callable[[str], None],
        config: SuggesterConfig,
    ) -> None:
        self._suggester = suggester
        self._on_suggestion = on_suggestion
        self._timeout = config.timeout_seconds
        self._recent: deque[Turn] = deque(maxlen=config.recent_turns)
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

    def cancel(self) -> None:
        """Abandons the Suggestion in progress."""
        if self._running is not None:
            self._running.cancel()
            self._running = None

    async def finish(self) -> None:
        """Waits for the Suggestion in progress, until its timeout at most."""
        if self._running is not None:
            await asyncio.wait([self._running])
        self._raise_output_failure()

    def _raise_output_failure(self) -> None:
        if self._output_failure is not None:
            raise self._output_failure

    async def _suggest(self, turns: Sequence[Turn]) -> None:
        started = time.monotonic()
        limit = asyncio.timeout(self._timeout)
        try:
            async with limit:
                suggestions = list(await self._suggester.suggest(turns))
        except Exception:
            # Replaced or abandoned: an error from being cancelled means nothing.
            if asyncio.current_task() is not self._running:
                return
            if limit.expired():
                logger.warning("Suggester timed out after %.1f s", self._timeout)
            else:
                logger.exception("Suggester failed")
            return
        # A Suggester that ignored being replaced must not show its stale answer.
        if asyncio.current_task() is not self._running:
            return
        if suggestions:
            logger.info("Suggester answered in %.2f s", time.monotonic() - started)
        try:
            for suggestion in suggestions:
                self._on_suggestion(suggestion)
        except Exception as e:  # noqa: BLE001 - re-raised by _raise_output_failure
            # Nothing awaits this task, so the next call to on_turn or finish raises it.
            self._output_failure = e

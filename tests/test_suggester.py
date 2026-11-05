import asyncio
import contextlib
import logging
from collections.abc import Callable, Sequence

import pytest

from rt_speech.config import SuggesterConfig
from rt_speech.suggester import DummySuggester, SuggestionTrigger, make_suggester
from rt_speech.turns import Speaker, Turn

TIMEOUT = 5.0


def client(text: str) -> Turn:
    return Turn(Speaker.CLIENT, text, 0.0, 1.0)


class FakeSuggester:
    """Answers with the text of the last Turn, and remembers what it was given."""

    def __init__(self) -> None:
        self.calls: list[list[Turn]] = []

    async def suggest(self, turns: Sequence[Turn]) -> list[str]:
        self.calls.append(list(turns))
        return [f"reply to {turns[-1].text}"]


def make_trigger(
    suggester: FakeSuggester, shown: list[str], recent_turns: int = 10
) -> SuggestionTrigger:
    return SuggestionTrigger(
        suggester,
        shown.append,
        SuggesterConfig(recent_turns=recent_turns, timeout_seconds=TIMEOUT),
    )


def test_a_closed_client_turn_produces_a_suggestion() -> None:
    suggester = FakeSuggester()
    shown: list[str] = []

    async def main() -> None:
        trigger = make_trigger(suggester, shown)
        trigger.on_turn(client("how much is it?"))
        await trigger.finish()

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == ["reply to how much is it?"]
    assert suggester.calls == [[client("how much is it?")]]


def me(text: str) -> Turn:
    return Turn(Speaker.ME, text, 0.0, 1.0)


def test_me_turns_add_context_but_never_trigger() -> None:
    suggester = FakeSuggester()
    shown: list[str] = []

    async def main() -> None:
        trigger = make_trigger(suggester, shown)
        trigger.on_turn(client("hello"))
        await trigger.finish()
        trigger.on_turn(me("hi there"))
        await trigger.finish()
        trigger.on_turn(client("how much is it?"))
        await trigger.finish()

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == ["reply to hello", "reply to how much is it?"]
    assert suggester.calls[-1] == [
        client("hello"),
        me("hi there"),
        client("how much is it?"),
    ]


def test_only_the_most_recent_turns_are_passed() -> None:
    suggester = FakeSuggester()

    async def main() -> None:
        trigger = make_trigger(suggester, [], recent_turns=2)
        for text in ["one", "two", "three"]:
            trigger.on_turn(client(text))
            await trigger.finish()

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert suggester.calls[-1] == [client("two"), client("three")]


class HeldSuggester(FakeSuggester):
    """Waits for `release` before answering, so a Suggestion can be caught in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def suggest(self, turns: Sequence[Turn]) -> list[str]:
        self.calls.append(list(turns))
        await self.release.wait()
        return [f"reply to {turns[-1].text}"]


def test_a_newer_client_turn_replaces_a_suggestion_still_in_progress() -> None:
    # An answer to what the Client said before is out of date once they've said more.
    shown: list[str] = []

    async def main() -> HeldSuggester:
        suggester = HeldSuggester()
        trigger = make_trigger(suggester, shown)
        trigger.on_turn(client("first"))
        await asyncio.sleep(0)
        trigger.on_turn(client("second"))
        await asyncio.sleep(0)
        suggester.release.set()
        await trigger.finish()
        return suggester

    suggester = asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == ["reply to second"]
    assert suggester.calls[-1] == [client("first"), client("second")]


def test_a_me_turn_leaves_a_suggestion_in_progress_alone() -> None:
    shown: list[str] = []

    async def main() -> None:
        suggester = HeldSuggester()
        trigger = make_trigger(suggester, shown)
        trigger.on_turn(client("first"))
        await asyncio.sleep(0)
        trigger.on_turn(me("thinking"))
        suggester.release.set()
        await trigger.finish()

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == ["reply to first"]


def test_a_failing_suggester_is_logged_and_later_turns_still_get_suggestions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailsOnce(FakeSuggester):
        async def suggest(self, turns: Sequence[Turn]) -> list[str]:
            if not self.calls:
                self.calls.append(list(turns))
                raise ConnectionError("service down")
            return await super().suggest(turns)

    shown: list[str] = []

    async def main() -> None:
        trigger = make_trigger(FailsOnce(), shown)
        trigger.on_turn(client("first"))
        await trigger.finish()
        trigger.on_turn(client("second"))
        await trigger.finish()

    with caplog.at_level(logging.ERROR):
        asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == ["reply to second"]
    assert "service down" in caplog.text


def test_a_suggester_that_never_answers_times_out_so_finish_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    shown: list[str] = []

    async def main() -> None:
        trigger = SuggestionTrigger(
            HeldSuggester(), shown.append, SuggesterConfig(timeout_seconds=0.01)
        )
        trigger.on_turn(client("first"))
        await trigger.finish()

    with caplog.at_level(logging.WARNING):
        asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == []
    assert "timed out" in caplog.text.lower()


def test_failing_output_is_raised_at_the_next_turn() -> None:
    # Printing to a closed pipe has to reach the CLI, the same as for a Turn.
    def closed_pipe(_: str) -> None:
        raise BrokenPipeError

    async def main() -> None:
        trigger = make_trigger_with_output(closed_pipe)
        trigger.on_turn(client("first"))
        await asyncio.sleep(0)
        with pytest.raises(BrokenPipeError):
            trigger.on_turn(me("second"))

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))


def test_failing_output_is_raised_by_finish() -> None:
    def closed_pipe(_: str) -> None:
        raise BrokenPipeError

    async def main() -> None:
        trigger = make_trigger_with_output(closed_pipe)
        trigger.on_turn(client("first"))
        with pytest.raises(BrokenPipeError):
            await trigger.finish()

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))


def make_trigger_with_output(output: Callable[[str], None]) -> SuggestionTrigger:
    return SuggestionTrigger(
        FakeSuggester(), output, SuggesterConfig(timeout_seconds=TIMEOUT)
    )


def test_the_dummy_answers_the_last_turn_and_says_it_is_a_dummy() -> None:
    turns = [client("hello"), me("hi"), client("how much is it?")]

    assert asyncio.run(DummySuggester().suggest(turns)) == [
        '(dummy) Follow up on "how much is it?"'
    ]


def test_the_dummy_stays_short_on_a_long_turn() -> None:
    words = " ".join(f"w{number}" for number in range(100))

    assert asyncio.run(DummySuggester().suggest([client(words)])) == [
        '(dummy) Follow up on "...w94 w95 w96 w97 w98 w99"'
    ]


def test_every_suggestion_is_shown_in_order() -> None:
    class Two(FakeSuggester):
        async def suggest(self, turns: Sequence[Turn]) -> list[str]:
            return ["one", "two"]

    shown: list[str] = []

    async def main() -> None:
        trigger = make_trigger(Two(), shown)
        trigger.on_turn(client("hello"))
        await trigger.finish()

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == ["one", "two"]


def test_a_replaced_suggestion_stays_hidden_even_if_it_ignores_the_cancel() -> None:
    class IgnoresCancel(FakeSuggester):
        async def suggest(self, turns: Sequence[Turn]) -> list[str]:
            first = not self.calls
            self.calls.append(list(turns))
            if first:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.Event().wait()
            return [f"reply to {turns[-1].text}"]

    shown: list[str] = []

    async def main() -> None:
        trigger = make_trigger(IgnoresCancel(), shown)
        trigger.on_turn(client("first"))
        await asyncio.sleep(0)
        trigger.on_turn(client("second"))
        await asyncio.sleep(0.01)
        await trigger.finish()

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == ["reply to second"]


def test_cancel_stops_the_suggestion_in_progress() -> None:
    shown: list[str] = []

    stopped: list[bool] = []

    class NotesCancel(FakeSuggester):
        async def suggest(self, turns: Sequence[Turn]) -> list[str]:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopped.append(True)
                raise
            return []

    async def main() -> None:
        trigger = make_trigger(NotesCancel(), shown)
        trigger.on_turn(client("first"))
        await asyncio.sleep(0)
        trigger.cancel()
        await asyncio.sleep(0.01)
        # Checked here, because ending the run cancels whatever is left anyway.
        assert stopped == [True]

    asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == []


def test_a_suggester_returning_nonsense_is_logged_as_a_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class ReturnsNone(FakeSuggester):
        async def suggest(self, turns: Sequence[Turn]) -> list[str]:
            return None  # type: ignore[return-value]

    async def main() -> None:
        trigger = make_trigger(ReturnsNone(), [])
        trigger.on_turn(client("first"))
        await trigger.finish()

    with caplog.at_level(logging.ERROR):
        asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert "Suggester failed" in caplog.text


def test_a_timeout_inside_the_suggester_is_reported_as_a_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A connection timing out says nothing about our own limit having passed.
    class ConnectTimeout(FakeSuggester):
        async def suggest(self, turns: Sequence[Turn]) -> list[str]:
            raise TimeoutError("connect timed out")

    async def main() -> None:
        trigger = make_trigger(ConnectTimeout(), [])
        trigger.on_turn(client("first"))
        await trigger.finish()

    with caplog.at_level(logging.ERROR):
        asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert "connect timed out" in caplog.text


def test_off_logs_nothing_and_shows_nothing(caplog: pytest.LogCaptureFixture) -> None:
    shown: list[str] = []

    async def main() -> None:
        config = SuggesterConfig(engine="off")
        trigger = SuggestionTrigger(make_suggester(config), shown.append, config)
        trigger.on_turn(client("first"))
        await trigger.finish()

    with caplog.at_level(logging.INFO):
        asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == []
    assert caplog.text == ""


def test_a_client_raising_its_own_error_when_cancelled_is_not_a_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The cancel was ours, so nothing went wrong.
    class ErrorOnCancel(FakeSuggester):
        async def suggest(self, turns: Sequence[Turn]) -> list[str]:
            if self.calls:
                return await super().suggest(turns)
            self.calls.append(list(turns))
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise ConnectionError("connection closed") from None
            return []

    shown: list[str] = []

    async def main() -> None:
        trigger = make_trigger(ErrorOnCancel(), shown)
        trigger.on_turn(client("first"))
        await asyncio.sleep(0)
        trigger.on_turn(client("second"))
        await trigger.finish()

    with caplog.at_level(logging.ERROR):
        asyncio.run(asyncio.wait_for(main(), TIMEOUT))

    assert shown == ["reply to second"]
    assert caplog.text == ""

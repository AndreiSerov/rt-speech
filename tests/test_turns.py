from rt_speech.audio import Stream
from rt_speech.turns import Speaker, Turn, TurnAggregator

PAUSE = 2.0


def aggregator() -> TurnAggregator:
    return TurnAggregator(pause_seconds=PAUSE)


def test_system_speech_is_the_client_and_mic_speech_is_me() -> None:
    turns = aggregator()

    turns.add(Stream.SYSTEM, "Здравствуйте", 0.0, 1.0)
    closed = turns.add(Stream.MIC, "Добрый день", 1.5, 2.0) + turns.flush()

    assert closed == [
        Turn(Speaker.CLIENT, "Здравствуйте", 0.0, 1.0),
        Turn(Speaker.ME, "Добрый день", 1.5, 2.0),
    ]


def test_consecutive_segments_of_a_speaker_join_one_turn() -> None:
    turns = aggregator()

    assert turns.add(Stream.SYSTEM, "Один", 0.0, 1.0) == []
    assert turns.add(Stream.SYSTEM, "два", 1.5, 2.0) == []

    assert turns.flush() == [Turn(Speaker.CLIENT, "Один два", 0.0, 2.0)]


def test_a_change_of_speaker_closes_the_turn() -> None:
    turns = aggregator()
    turns.add(Stream.SYSTEM, "Вопрос", 0.0, 1.0)

    assert turns.add(Stream.MIC, "Ответ", 1.1, 2.0) == [
        Turn(Speaker.CLIENT, "Вопрос", 0.0, 1.0)
    ]


def test_a_long_gap_between_segments_closes_the_turn() -> None:
    turns = aggregator()
    turns.add(Stream.SYSTEM, "Один", 0.0, 1.0)

    assert turns.add(Stream.SYSTEM, "два", 3.0, 4.0) == [
        Turn(Speaker.CLIENT, "Один", 0.0, 1.0)
    ]
    assert turns.flush() == [Turn(Speaker.CLIENT, "два", 3.0, 4.0)]


def test_the_clock_closes_a_turn_after_the_pause() -> None:
    turns = aggregator()
    turns.add(Stream.SYSTEM, "Вопрос", 0.0, 1.0)

    assert turns.clock(2.9, speaking=frozenset()) == []
    assert turns.clock(3.0, speaking=frozenset()) == [
        Turn(Speaker.CLIENT, "Вопрос", 0.0, 1.0)
    ]
    assert turns.flush() == []


def test_the_clock_waits_while_the_same_stream_is_mid_speech() -> None:
    # The Client resumed at 2.5 s, but that Segment hasn't finished yet.
    turns = aggregator()
    turns.add(Stream.SYSTEM, "Один", 0.0, 1.0)

    assert turns.clock(3.5, speaking=frozenset({Stream.SYSTEM})) == []

    turns.add(Stream.SYSTEM, "два", 2.5, 4.0)
    assert turns.flush() == [Turn(Speaker.CLIENT, "Один два", 0.0, 4.0)]


def test_the_other_stream_speaking_does_not_hold_the_turn_open() -> None:
    turns = aggregator()
    turns.add(Stream.SYSTEM, "Вопрос", 0.0, 1.0)

    assert turns.clock(3.0, speaking=frozenset({Stream.MIC})) == [
        Turn(Speaker.CLIENT, "Вопрос", 0.0, 1.0)
    ]


def test_empty_text_does_not_close_or_extend_a_turn() -> None:
    # A Segment whose text was filtered out must not interrupt the Client.
    turns = aggregator()
    turns.add(Stream.SYSTEM, "Вопрос", 0.0, 1.0)

    assert turns.add(Stream.MIC, "", 1.1, 1.5) == []

    assert turns.flush() == [Turn(Speaker.CLIENT, "Вопрос", 0.0, 1.0)]


def test_flush_with_nothing_open_produces_nothing() -> None:
    assert aggregator().flush() == []


def test_the_clock_with_nothing_open_produces_nothing() -> None:
    assert aggregator().clock(100.0, speaking=frozenset()) == []


def test_finish_order_is_kept_when_both_talk_at_once() -> None:
    # "Угу" at 5 s finishes before the Client's 0-10 s Segment does.
    turns = aggregator()

    turns.add(Stream.MIC, "Угу", 5.0, 5.5)
    closed = turns.add(Stream.SYSTEM, "Длинная фраза", 0.0, 10.0) + turns.flush()

    assert closed == [
        Turn(Speaker.ME, "Угу", 5.0, 5.5),
        Turn(Speaker.CLIENT, "Длинная фраза", 0.0, 10.0),
    ]

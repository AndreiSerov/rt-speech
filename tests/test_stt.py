import logging

import numpy as np
import pytest

from rt_speech.config import SttFilterConfig
from rt_speech.stt import HallucinationFilter, TranscriptPiece

LOUD_AUDIO = np.full(16_000, 0.2, dtype=np.float32)
NEAR_SILENCE = np.full(16_000, 0.001, dtype=np.float32)

FILTER = HallucinationFilter(
    SttFilterConfig(
        no_speech_threshold=0.6,
        avg_logprob_threshold=-1.0,
        near_silence_rms=0.01,
        filler_phrases=["Спасибо за просмотр", "thanks for watching"],
    )
)


def test_confident_text_is_kept() -> None:
    pieces = [TranscriptPiece("Добрый день", no_speech_prob=0.1, avg_logprob=-0.2)]

    assert FILTER.keep(pieces, LOUD_AUDIO) == "Добрый день"


def test_pieces_join_with_trimmed_whitespace() -> None:
    pieces = [TranscriptPiece(" Добрый день."), TranscriptPiece(" Как дела? ")]

    assert FILTER.keep(pieces, LOUD_AUDIO) == "Добрый день. Как дела?"


def test_high_no_speech_with_low_confidence_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pieces = [TranscriptPiece("Шум", no_speech_prob=0.9, avg_logprob=-2.0)]

    with caplog.at_level(logging.INFO):
        assert FILTER.keep(pieces, LOUD_AUDIO) == ""

    assert "Шум" in caplog.text
    assert "no_speech_prob" in caplog.text


@pytest.mark.parametrize(
    ("no_speech", "logprob"),
    [(0.9, -0.2), (0.1, -2.0)],
    ids=["only-no-speech-high", "only-confidence-low"],
)
def test_one_bad_value_alone_keeps_the_text(no_speech: float, logprob: float) -> None:
    pieces = [TranscriptPiece("Тихо", no_speech_prob=no_speech, avg_logprob=logprob)]

    assert FILTER.keep(pieces, LOUD_AUDIO) == "Тихо"


def test_an_engine_without_confidence_values_is_not_filtered_on_them() -> None:
    assert FILTER.keep([TranscriptPiece("Текст")], LOUD_AUDIO) == "Текст"


def test_one_bad_piece_does_not_drop_its_neighbours() -> None:
    pieces = [
        TranscriptPiece("Хорошо", no_speech_prob=0.1, avg_logprob=-0.2),
        TranscriptPiece("Шум", no_speech_prob=0.9, avg_logprob=-2.0),
    ]

    assert FILTER.keep(pieces, LOUD_AUDIO) == "Хорошо"


@pytest.mark.parametrize(
    "text", ["Спасибо за просмотр", " спасибо за просмотр! ", "Thanks for watching."]
)
def test_filler_phrase_on_near_silence_is_dropped(text: str) -> None:
    assert FILTER.keep([TranscriptPiece(text)], NEAR_SILENCE) == ""


def test_filler_phrase_on_real_audio_is_kept() -> None:
    text = "Спасибо за просмотр"

    assert FILTER.keep([TranscriptPiece(text)], LOUD_AUDIO) == text


def test_other_text_on_near_silence_is_kept() -> None:
    assert FILTER.keep([TranscriptPiece("Да")], NEAR_SILENCE) == "Да"


def test_empty_pieces_produce_no_text() -> None:
    assert FILTER.keep([TranscriptPiece("  ")], LOUD_AUDIO) == ""


@pytest.mark.parametrize(
    "text",
    [
        "“Thank you”",
        "♪ Thank you ♪",
        "¡Thank you!",
        "Thanks, for watching",
        "Спасибо,  за\xa0просмотр",
    ],
)
def test_filler_matching_ignores_punctuation_and_spacing(text: str) -> None:
    config = SttFilterConfig(
        filler_phrases=["Thank you", "Thanks for watching", "Спасибо за просмотр"]
    )

    assert HallucinationFilter(config).keep([TranscriptPiece(text)], NEAR_SILENCE) == ""


@pytest.mark.parametrize("text", ["...", "…", "—", "♪ ♪"])
def test_text_with_no_words_is_dropped_on_any_audio(text: str) -> None:
    assert FILTER.keep([TranscriptPiece(text)], LOUD_AUDIO) == ""


@pytest.mark.parametrize(
    ("no_speech", "logprob"),
    [(0.9, None), (None, -2.0)],
    ids=["only-no-speech", "only-logprob"],
)
def test_a_piece_with_only_one_value_is_not_dropped_on_it(
    no_speech: float | None, logprob: float | None
) -> None:
    piece = TranscriptPiece("Текст", no_speech_prob=no_speech, avg_logprob=logprob)

    assert FILTER.keep([piece], LOUD_AUDIO) == "Текст"


def test_values_exactly_on_the_thresholds_are_kept() -> None:
    piece = TranscriptPiece("Текст", no_speech_prob=0.6, avg_logprob=-1.0)

    assert FILTER.keep([piece], LOUD_AUDIO) == "Текст"


def test_empty_audio_counts_as_near_silence() -> None:
    empty = np.zeros(0, dtype=np.float32)

    assert FILTER.keep([TranscriptPiece("Thanks for watching")], empty) == ""

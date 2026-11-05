import logging
from collections.abc import Iterator

import pytest

from rt_speech.logging_setup import setup_logging


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_logs_go_to_stderr_and_not_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")

    logging.getLogger("test").info("hello")

    captured = capsys.readouterr()
    assert "hello" in captured.err
    assert captured.out == ""


def test_level_filters_messages(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("WARNING")

    logging.getLogger("test").info("hidden")

    assert "hidden" not in capsys.readouterr().err

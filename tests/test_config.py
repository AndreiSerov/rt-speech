from pathlib import Path

import pytest

from rt_speech.config import Config, ConfigError, load_config

REPO_ROOT = Path(__file__).parent.parent


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content)
    return path


def test_values_are_loaded(tmp_path: Path) -> None:
    path = write(tmp_path, '[logging]\nlevel = "DEBUG"\n[replay]\nrealtime = true\n')

    config = load_config(path)

    assert config.logging.level == "DEBUG"
    assert config.replay.realtime is True


def test_missing_values_use_defaults(tmp_path: Path) -> None:
    assert load_config(write(tmp_path, "")) == Config()


def test_the_committed_config_is_valid() -> None:
    load_config(REPO_ROOT / "config.toml")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('[logging]\nlevle = "INFO"\n', "levle"),
        ("[unknown_section]\nx = 1\n", "unknown_section"),
        ('[logging]\nlevel = "LOUD"\n', "level"),
        ('[replay]\nrealtime = "maybe"\n', "realtime"),
        ("[logging\n", "config.toml"),
        ("[segmenter]\nsilence_ende_ms = 600\n", "silence_ende_ms"),
        ("[segmenter]\nvoice_threshold = 1.5\n", "voice_threshold"),
        ("[segmenter]\nsplit_search_ms = 20000\n", "split_search_ms"),
        ("[segmenter]\nsplit_window_ms = 3000\n", "split_window_ms"),
        ("[segmenter]\npreroll_ms = 18000\n", "preroll_ms"),
        ("[segmenter]\nmin_speech_ms = 17000\n", "min_speech_ms"),
        ("[segmenter]\nsilence_end_ms = 18000\n", "silence_end_ms"),
        ("[segmenter]\nspeech_start_frames = 600\n", "speech_start_frames"),
        ('[suggester]\nengine = "gpt"\n', "engine"),
        ("[suggester]\nrecent_turns = 0\n", "recent_turns"),
        ("[suggester]\ntimeout_seconds = 0\n", "timeout_seconds"),
    ],
)
def test_bad_config_is_rejected(tmp_path: Path, content: str, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_config(write(tmp_path, content))


def test_suggester_settings_are_loaded(tmp_path: Path) -> None:
    path = write(
        tmp_path, '[suggester]\nengine = "off"\nrecent_turns = 3\ntimeout_seconds = 2.5\n'
    )

    suggester = load_config(path).suggester

    assert (suggester.engine, suggester.recent_turns, suggester.timeout_seconds) == (
        "off",
        3,
        2.5,
    )


def test_a_non_utf8_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(ConfigError, match="config.toml"):
        load_config(path)


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")

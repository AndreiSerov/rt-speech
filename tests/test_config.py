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
    ],
)
def test_bad_config_is_rejected(tmp_path: Path, content: str, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_config(write(tmp_path, content))


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")

"""Loads and validates the TOML config file."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class _Section(BaseModel):
    # Unknown keys are errors: a mistyped threshold must not silently use its default.
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ReplayConfig(_Section):
    realtime: bool = False


class Config(_Section):
    logging: LoggingConfig = LoggingConfig()
    replay: ReplayConfig = ReplayConfig()


class ConfigError(Exception):
    """The config file is missing, malformed, or has invalid values."""


def load_config(path: Path) -> Config:
    try:
        with path.open("rb") as f:
            return Config.model_validate(tomllib.load(f))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except (tomllib.TOMLDecodeError, ValidationError) as e:
        raise ConfigError(f"{path}: {e}") from e

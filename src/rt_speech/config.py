"""Loads and validates the TOML config file."""

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE


class _Section(BaseModel):
    # Unknown keys are errors: a mistyped threshold must not silently use its default.
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ReplayConfig(_Section):
    realtime: bool = False


class SegmenterConfig(_Section):
    voice_threshold: float = Field(default=0.5, gt=0, lt=1)
    speech_start_frames: int = Field(default=3, gt=0)
    silence_end_ms: int = Field(default=600, gt=0)
    preroll_ms: int = Field(default=300, ge=0)
    postroll_ms: int = Field(default=200, ge=0)
    min_speech_ms: int = Field(default=300, ge=0)
    max_segment_ms: int = Field(default=18_000, gt=0)
    split_search_ms: int = Field(default=2_000, gt=0)
    split_window_ms: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def _check_cap_fits(self) -> Self:
        if self.split_window_ms > self.split_search_ms:
            raise ValueError("split_window_ms can't exceed split_search_ms")
        # A cut must never land before enough speech to pass the minimum.
        start_ms = self.speech_start_frames * FRAME_SAMPLES * 1000 / SAMPLE_RATE
        needed = self.preroll_ms + start_ms + self.min_speech_ms + self.split_search_ms
        if needed > self.max_segment_ms:
            raise ValueError(
                "preroll_ms + speech_start_frames + min_speech_ms + split_search_ms "
                "must fit within max_segment_ms"
            )
        return self


class Config(_Section):
    logging: LoggingConfig = LoggingConfig()
    replay: ReplayConfig = ReplayConfig()
    segmenter: SegmenterConfig = SegmenterConfig()


class ConfigError(Exception):
    """The config file is missing, malformed, or has invalid values."""


def load_config(path: Path) -> Config:
    try:
        with path.open("rb") as f:
            return Config.model_validate(tomllib.load(f))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError) as e:
        raise ConfigError(f"{path}: {e}") from e

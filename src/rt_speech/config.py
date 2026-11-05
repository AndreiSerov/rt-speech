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
        if self.silence_end_ms >= self.max_segment_ms:
            raise ValueError("silence_end_ms must be below max_segment_ms")
        # A cut must never land before enough speech to pass the minimum.
        start_ms = self.speech_start_frames * FRAME_SAMPLES * 1000 / SAMPLE_RATE
        needed = self.preroll_ms + start_ms + self.min_speech_ms + self.split_search_ms
        if needed > self.max_segment_ms:
            raise ValueError(
                "preroll_ms + speech_start_frames + min_speech_ms + split_search_ms "
                "must fit within max_segment_ms"
            )
        return self


Language = Literal["ru", "en", "auto"]


class SttFilterConfig(_Section):
    no_speech_threshold: float = Field(default=0.6, ge=0, le=1)
    avg_logprob_threshold: float = Field(default=-1.0, le=0)
    near_silence_rms: float = Field(default=0.01, ge=0)
    filler_phrases: list[str] = [
        "Спасибо за просмотр",
        "Субтитры сделал DimaTorzok",
        "Продолжение следует",
        "Thanks for watching",
        "Thank you",
        "You",
    ]


class MlxWhisperConfig(_Section):
    model: str = "mlx-community/whisper-large-v3-turbo"
    filter: SttFilterConfig = SttFilterConfig()


class FasterWhisperConfig(_Section):
    model: str = "large-v3-turbo"
    compute_type: str = "int8"
    filter: SttFilterConfig = SttFilterConfig()


class SttConfig(_Section):
    engine: Literal["mlx-whisper", "faster-whisper"] = "mlx-whisper"
    language: Language = "ru"
    system_language: Language | None = None
    mic_language: Language | None = None
    queue_warning_segments: int = Field(default=5, gt=0)
    mlx_whisper: MlxWhisperConfig = MlxWhisperConfig()
    faster_whisper: FasterWhisperConfig = FasterWhisperConfig()


class TurnsConfig(_Section):
    pause_ms: int = Field(default=1_800, gt=0)


class CaptureConfig(_Section):
    binary: Path = Path("native/capture/.build/release/rt-capture")


class Config(_Section):
    capture: CaptureConfig = CaptureConfig()
    logging: LoggingConfig = LoggingConfig()
    replay: ReplayConfig = ReplayConfig()
    segmenter: SegmenterConfig = SegmenterConfig()
    stt: SttConfig = SttConfig()
    turns: TurnsConfig = TurnsConfig()


class ConfigError(Exception):
    """The config file is missing, malformed, or has invalid values."""


def load_config(path: Path) -> Config:
    try:
        with path.open("rb") as f:
            return Config.model_validate(tomllib.load(f))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError) as e:
        raise ConfigError(f"{path}: {e}") from e

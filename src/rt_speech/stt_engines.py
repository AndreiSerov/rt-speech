"""The real speech-to-text engines, each an optional install imported on demand."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rt_speech.audio import SAMPLE_RATE
from rt_speech.config import FasterWhisperConfig, MlxWhisperConfig, SttConfig
from rt_speech.stt import Transcriber, TranscriptPiece


class SttSetupError(Exception):
    """An engine isn't installed, or its model can't be loaded."""


class _MlxWhisper:
    def __init__(self, config: MlxWhisperConfig) -> None:
        import mlx_whisper

        self._transcribe = mlx_whisper.transcribe
        self._model = config.model
        self._filter = config.filter

    def transcribe(
        self, pcm: NDArray[np.float32], language: str | None
    ) -> list[TranscriptPiece]:
        result: dict[str, Any] = self._transcribe(
            pcm,
            path_or_hf_repo=self._model,
            language=language,
            condition_on_previous_text=False,
            # Same limits as our filter, which lets the engine skip silence early
            # rather than retrying it at every temperature.
            no_speech_threshold=self._filter.no_speech_threshold,
            logprob_threshold=self._filter.avg_logprob_threshold,
        )
        return [
            TranscriptPiece(s["text"], s["no_speech_prob"], s["avg_logprob"])
            for s in result["segments"]
        ]


class _FasterWhisper:
    def __init__(self, config: FasterWhisperConfig) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            config.model, device="cpu", compute_type=config.compute_type
        )
        self._filter = config.filter

    def transcribe(
        self, pcm: NDArray[np.float32], language: str | None
    ) -> list[TranscriptPiece]:
        segments, _ = self._model.transcribe(
            pcm,
            language=language,
            condition_on_previous_text=False,
            no_speech_threshold=self._filter.no_speech_threshold,
            log_prob_threshold=self._filter.avg_logprob_threshold,
        )
        return [
            TranscriptPiece(s.text, s.no_speech_prob, s.avg_logprob) for s in segments
        ]


def make_transcriber(config: SttConfig) -> Transcriber:
    """Builds the configured engine and runs it once, so a missing install or model
    fails at startup, and the first real Segment doesn't pay for loading the model."""
    extra = "mlx" if config.engine == "mlx-whisper" else "faster"
    try:
        engine: Transcriber = (
            _MlxWhisper(config.mlx_whisper)
            if config.engine == "mlx-whisper"
            else _FasterWhisper(config.faster_whisper)
        )
    except ImportError as e:
        raise SttSetupError(
            f"{config.engine} isn't installed. Run: uv sync --extra {extra}"
        ) from e
    except Exception as e:
        raise SttSetupError(f"can't load {config.engine}: {e}") from e
    try:
        engine.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), "en")
    except Exception as e:
        model = (
            config.mlx_whisper.model
            if config.engine == "mlx-whisper"
            else config.faster_whisper.model
        )
        raise SttSetupError(f"can't run {config.engine} with {model}: {e}") from e
    return engine

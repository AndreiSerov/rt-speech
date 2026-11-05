import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rt_speech.audio import SAMPLE_RATE
from rt_speech.config import (
    FasterWhisperConfig,
    MlxWhisperConfig,
    SttConfig,
    SttFilterConfig,
)
from rt_speech.stt import HallucinationFilter
from rt_speech.stt_engines import SttSetupError, make_transcriber
from tests.conftest import read_frames

ENGINES = ["mlx-whisper", "faster-whisper"]


@pytest.mark.parametrize(
    ("engine", "module", "extra"),
    [("mlx-whisper", "mlx_whisper", "mlx"), ("faster-whisper", "faster_whisper", "faster")],
)
def test_a_missing_engine_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch, engine: str, module: str, extra: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(SttSetupError, match=f"uv sync --extra {extra}"):
        make_transcriber(SttConfig(engine=engine))


def test_mlx_engine_gets_the_configured_thresholds_and_no_previous_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def fake_transcribe(pcm: np.ndarray, **kwargs: object) -> dict:
        calls.append(kwargs)
        return {"segments": []}

    monkeypatch.setitem(
        sys.modules, "mlx_whisper", SimpleNamespace(transcribe=fake_transcribe)
    )
    config = SttConfig(
        mlx_whisper=MlxWhisperConfig(
            model="m",
            filter=SttFilterConfig(no_speech_threshold=0.7, avg_logprob_threshold=-2.0),
        )
    )

    make_transcriber(config).transcribe(np.zeros(10, dtype=np.float32), "ru")

    assert calls[-1] == {
        "path_or_hf_repo": "m",
        "language": "ru",
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.7,
        "logprob_threshold": -2.0,
    }


def test_faster_engine_gets_the_configured_thresholds_and_no_previous_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class FakeModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            pass

        def transcribe(self, pcm: np.ndarray, **kwargs: object) -> tuple[list, None]:
            calls.append(kwargs)
            return [], None

    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeModel)
    )
    config = SttConfig(
        engine="faster-whisper",
        faster_whisper=FasterWhisperConfig(
            filter=SttFilterConfig(no_speech_threshold=0.7, avg_logprob_threshold=-2.0),
        ),
    )

    make_transcriber(config).transcribe(np.zeros(10, dtype=np.float32), None)

    assert calls[-1] == {
        "language": None,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.7,
        "log_prob_threshold": -2.0,
    }


def test_a_model_that_cant_load_is_a_setup_error(tmp_path: Path) -> None:
    pytest.importorskip("mlx_whisper")
    config = SttConfig(mlx_whisper=MlxWhisperConfig(model=str(tmp_path / "no-model")))

    with pytest.raises(SttSetupError, match="no-model"):
        make_transcriber(config)


def _audio(path: Path) -> np.ndarray:
    return np.concatenate(read_frames(path))


@pytest.mark.manual
@pytest.mark.parametrize("engine", ENGINES)
def test_real_engine_transcribes_english(engine: str, speech_wav: Path) -> None:
    transcriber = make_transcriber(SttConfig(engine=engine))

    pieces = transcriber.transcribe(_audio(speech_wav), "en")

    text = " ".join(p.text for p in pieces).lower()
    assert "test" in text and "voice" in text
    assert all(p.no_speech_prob is not None for p in pieces)
    assert all(p.avg_logprob is not None for p in pieces)


@pytest.mark.manual
@pytest.mark.parametrize("engine", ENGINES)
def test_real_engine_transcribes_russian(engine: str, russian_speech_wav: Path) -> None:
    transcriber = make_transcriber(SttConfig(engine=engine))

    pieces = transcriber.transcribe(_audio(russian_speech_wav), "ru")

    text = " ".join(p.text for p in pieces).lower()
    assert "здравствуйте" in text and "проверка" in text


@pytest.mark.manual
@pytest.mark.parametrize("engine", ENGINES)
def test_real_engine_detects_the_language_when_none_is_given(
    engine: str, russian_speech_wav: Path
) -> None:
    transcriber = make_transcriber(SttConfig(engine=engine))

    pieces = transcriber.transcribe(_audio(russian_speech_wav), None)

    assert "здравствуйте" in " ".join(p.text for p in pieces).lower()


@pytest.mark.manual
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("language", ["ru", "en"])
def test_real_engine_hallucinations_on_silence_are_filtered(
    engine: str, language: str
) -> None:
    # large-v3-turbo invents confident text on silence with no_speech_prob 0.0, so
    # only the filler phrases can catch it.
    config = SttConfig(engine=engine)
    transcriber = make_transcriber(config)
    silence = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    engine_config = (
        config.mlx_whisper if engine == "mlx-whisper" else config.faster_whisper
    )

    pieces = transcriber.transcribe(silence, language)

    assert HallucinationFilter(engine_config.filter).keep(pieces, silence) == ""

import sys
from pathlib import Path

import numpy as np
import pytest

from rt_speech.audio import SAMPLE_RATE
from rt_speech.config import MlxWhisperConfig, SttConfig
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
def test_real_engine_reports_silence_as_no_speech(engine: str) -> None:
    transcriber = make_transcriber(SttConfig(engine=engine))

    pieces = transcriber.transcribe(np.zeros(SAMPLE_RATE * 3, dtype=np.float32), "ru")

    assert all(p.no_speech_prob is None or p.no_speech_prob > 0.3 for p in pieces)

"""Speech-to-text: the engine interface and the hallucination filter."""

import logging
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rt_speech.config import SttFilterConfig

logger = logging.getLogger(__name__)

_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class TranscriptPiece:
    """One span of text from an engine. Engines that lack a confidence value leave
    it as None, and the filter then doesn't judge the piece on it."""

    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


class Transcriber(Protocol):
    def transcribe(
        self, pcm: NDArray[np.float32], language: str | None
    ) -> list[TranscriptPiece]:
        """Transcribes a Segment's audio. A language of None means auto-detect.
        May raise; the pipeline logs the failure and moves on."""
        ...


def _words(text: str) -> str:
    """The text with case, punctuation and spacing removed, for comparing phrases."""
    return " ".join(_WORD.findall(text)).casefold()


class HallucinationFilter:
    def __init__(self, config: SttFilterConfig) -> None:
        self._config = config
        self._fillers = {_words(phrase) for phrase in config.filler_phrases}

    def keep(self, pieces: list[TranscriptPiece], pcm: NDArray[np.float32]) -> str:
        """Returns the text worth keeping from one Segment, possibly empty."""
        rms = float(np.sqrt(np.mean(pcm**2))) if pcm.size else 0.0
        kept = []
        for piece in pieces:
            text = piece.text.strip()
            reason = self._drop_reason(piece, text, rms)
            if reason:
                logger.info("Dropped %r: %s", text, reason)
            else:
                kept.append(text)
        return " ".join(kept)

    def _drop_reason(self, piece: TranscriptPiece, text: str, rms: float) -> str | None:
        words = _words(text)
        if not words:
            return "no words"
        no_speech, logprob = piece.no_speech_prob, piece.avg_logprob
        config = self._config
        if (
            no_speech is not None
            and logprob is not None
            and no_speech > config.no_speech_threshold
            and logprob < config.avg_logprob_threshold
        ):
            return f"no_speech_prob {no_speech:.2f}, avg_logprob {logprob:.2f}"
        if rms < config.near_silence_rms and words in self._fillers:
            return f"filler phrase on near-silence (rms {rms:.4f})"
        return None

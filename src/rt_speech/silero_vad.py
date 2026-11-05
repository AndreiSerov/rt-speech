"""Silero VAD run directly through onnxruntime, without torch."""

from importlib import resources

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from rt_speech.audio import FRAME_SAMPLES, SAMPLE_RATE

# The model expects the end of the previous frame prepended to each new one.
_CONTEXT_SAMPLES = 64
_SAMPLE_RATE_INPUT = np.array(SAMPLE_RATE, dtype=np.int64)


class SileroVad:
    """Scores consecutive frames of one Stream. The model is recurrent, so never
    share an instance between Streams."""

    def __init__(self) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        model = resources.files("rt_speech.assets").joinpath("silero_vad.onnx")
        self._session = ort.InferenceSession(
            model.read_bytes(), options, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SAMPLES), dtype=np.float32)

    def speech_probability(self, pcm: NDArray[np.float32]) -> float:
        if pcm.shape != (FRAME_SAMPLES,) or pcm.dtype != np.float32:
            raise ValueError(
                f"expected {FRAME_SAMPLES} float32 samples, "
                f"got shape {pcm.shape} of {pcm.dtype}"
            )
        frame = pcm.reshape(1, -1)
        output, self._state = self._session.run(
            None,
            {
                "input": np.concatenate([self._context, frame], axis=1),
                "state": self._state,
                "sr": _SAMPLE_RATE_INPUT,
            },
        )
        # Copied, because the caller may reuse its buffer for the next frame.
        self._context = frame[:, -_CONTEXT_SAMPLES:].copy()
        return float(output[0, 0])

"""
Speech-to-text using NVIDIA Parakeet TDT 1.1b (NeMo, CUDA).
English-only, faster and more accurate than Whisper for this use case.
"""
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

PARAKEET_MODEL = os.getenv("PARAKEET_MODEL", "nvidia/parakeet-tdt-1.1b")

_model = None


def load() -> None:
    global _model
    if _model is not None:
        return
    import nemo.collections.asr as nemo_asr
    logger.info("Loading Parakeet model %s...", PARAKEET_MODEL)
    _model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL)
    _model.cuda()
    _model.eval()
    logger.info("Parakeet model loaded.")


def transcribe(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """
    Transcribe raw 16-bit mono PCM audio bytes to text.
    Returns empty string if audio is silent or too short.
    """
    if _model is None:
        load()

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio) < sample_rate * 0.3:  # less than 300ms — skip
        return ""

    # NeMo transcribe accepts a list of numpy arrays directly
    results = _model.transcribe([audio], batch_size=1)

    # Results is a list of strings (or hypotheses objects depending on version)
    if not results:
        return ""
    result = results[0]
    return (result.text if hasattr(result, "text") else str(result)).strip()

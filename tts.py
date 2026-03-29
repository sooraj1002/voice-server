"""
Text-to-speech using Kokoro ONNX (fast, 82M params).
Falls back to Piper TTS if Kokoro is unavailable.
"""
import io
import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
KOKORO_ONNX_PATH = os.getenv("KOKORO_ONNX_PATH", "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "voices-v1.0.bin")
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", "")

_kokoro = None
_use_piper = False


def load() -> None:
    global _kokoro, _use_piper
    if _kokoro is not None:
        return

    if not os.path.exists(KOKORO_ONNX_PATH):
        logger.warning(
            "Kokoro ONNX model not found at %s. Falling back to Piper.", KOKORO_ONNX_PATH
        )
        _use_piper = True
        return

    try:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(KOKORO_ONNX_PATH, KOKORO_VOICES_PATH)
        logger.info("Kokoro TTS loaded (voice=%s).", KOKORO_VOICE)
    except Exception as e:
        logger.warning("Failed to load Kokoro: %s. Falling back to Piper.", e)
        _use_piper = True


def synthesize(text: str) -> bytes:
    """
    Convert text to speech. Returns WAV bytes (PCM 24kHz).
    """
    if _kokoro is None and not _use_piper:
        load()

    text = text.strip()
    if not text:
        return b""

    if _use_piper:
        return _synthesize_piper(text)
    return _synthesize_kokoro(text)


def _synthesize_kokoro(text: str) -> bytes:
    import numpy as np
    import soundfile as sf

    # Split on sentence boundaries for better prosody
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text]

    all_samples = []
    sample_rate = 24000

    for sentence in sentences:
        samples, sr = _kokoro.create(sentence, voice=KOKORO_VOICE, speed=1.0, lang="en-us")
        sample_rate = sr
        all_samples.append(samples)

    import numpy as np
    combined = np.concatenate(all_samples) if len(all_samples) > 1 else all_samples[0]

    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _synthesize_piper(text: str) -> bytes:
    if not PIPER_MODEL_PATH:
        logger.error("Piper model path not set (PIPER_MODEL_PATH env var).")
        return b""
    try:
        proc = subprocess.run(
            ["piper", "--model", PIPER_MODEL_PATH, "--output_file", "-"],
            input=text.encode(),
            capture_output=True,
            timeout=30,
        )
        return proc.stdout
    except Exception as e:
        logger.error("Piper synthesis failed: %s", e)
        return b""

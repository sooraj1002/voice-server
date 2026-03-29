"""
Manages the lifecycle of the Ollama model — load on demand, unload after inactivity.
"""
import asyncio
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("VOICE_MODEL", "qwen3:8b")
INACTIVITY_TIMEOUT = int(os.getenv("INACTIVITY_TIMEOUT", "600"))


class ModelManager:
    def __init__(self) -> None:
        self._loaded = False
        self._lock = asyncio.Lock()
        self._last_activity: float | None = None
        self._watchdog_task: asyncio.Task | None = None

    async def start_watchdog(self) -> None:
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("Model watchdog started (timeout=%ds)", INACTIVITY_TIMEOUT)

    async def ensure_loaded(self) -> None:
        async with self._lock:
            if self._loaded:
                return
            logger.info("Loading model %s into Ollama...", MODEL_NAME)
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": MODEL_NAME, "prompt": "", "keep_alive": -1},
                ) as resp:
                    resp.raise_for_status()
                    # Drain the stream — first chunk confirms the model is in VRAM
                    async for _ in resp.aiter_bytes():
                        break
            self._loaded = True
            self.touch()
            logger.info("Model %s loaded.", MODEL_NAME)

    def touch(self) -> None:
        self._last_activity = time.monotonic()

    async def unload(self) -> None:
        async with self._lock:
            if not self._loaded:
                return
            logger.info("Unloading model %s from VRAM...", MODEL_NAME)
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": MODEL_NAME, "prompt": "", "keep_alive": 0},
                )
            self._loaded = False
            logger.info("Model unloaded.")

    async def get_status(self) -> dict:
        return {
            "loaded": self._loaded,
            "model": MODEL_NAME,
            "last_activity": self._last_activity,
            "inactivity_timeout": INACTIVITY_TIMEOUT,
        }

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            if (
                self._loaded
                and self._last_activity is not None
                and time.monotonic() - self._last_activity > INACTIVITY_TIMEOUT
            ):
                logger.info("Inactivity timeout reached, unloading model.")
                await self.unload()


model_manager = ModelManager()

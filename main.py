"""
FastAPI voice server — WebSocket audio pipeline + wake/status endpoints.
"""
import json
import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

import stt
import tts
from llm import build_system_prompt
from model_manager import model_manager
from pipeline import process_turn
from tools import TOOL_DEFINITIONS, prefetch_session_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting voice server...")
    await model_manager.start_watchdog()
    stt.load()
    tts.load()
    logger.info("Voice server ready.")
    yield
    logger.info("Voice server shutting down.")


app = FastAPI(title="Expense Tracker Voice Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip() or None
    return None


@app.post("/wake")
async def wake(request: Request):
    """
    Load the LLM model into VRAM. Called by the browser when voice mode opens.
    Blocks until the model is confirmed loaded (~10-20s first time).
    """
    token = _extract_bearer(request)
    if not token:
        raise HTTPException(status_code=401, detail="Authorization header required.")
    await model_manager.ensure_loaded()
    return {"status": "ready", "model": os.getenv("VOICE_MODEL", "qwen3:8b")}


@app.get("/status")
async def status():
    """Return model load state and last activity timestamp."""
    return await model_manager.get_status()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Voice session WebSocket.

    Protocol (client → server):
      - binary frames: WebM/Opus audio chunks
      - text frame:    {"type": "end_turn"}   — process buffered audio
      - text frame:    {"type": "cancel"}      — discard buffer, go back to listening

    Protocol (server → client):
      - text:   {"type": "thinking"}
      - text:   {"type": "transcript", "text": "..."}
      - binary: WAV audio bytes (TTS)
      - text:   {"type": "done"}
      - text:   {"type": "tool_result", "tool": "add_expense"}
      - text:   {"type": "error", "message": "..."}
    """
    token = websocket.query_params.get("token", "").strip()
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    # Prefetch categories & accounts for this session
    session_cache = await prefetch_session_data(token)

    # Build conversation history with system prompt
    conversation_history: list[dict] = [
        {"role": "system", "content": build_system_prompt()}
    ]

    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                audio_buffer.extend(message["bytes"])

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "cancel":
                    audio_buffer.clear()

                elif msg_type == "end_turn":
                    if not audio_buffer:
                        continue

                    await websocket.send_text(json.dumps({"type": "thinking"}))

                    try:
                        transcript, wav_bytes, action = await process_turn(
                            bytes(audio_buffer),
                            token,
                            conversation_history,
                            session_cache,
                        )
                        audio_buffer.clear()

                        if transcript:
                            await websocket.send_text(
                                json.dumps({"type": "transcript", "text": transcript})
                            )

                        if wav_bytes:
                            await websocket.send_bytes(wav_bytes)

                        if action:
                            await websocket.send_text(json.dumps({
                                "type": "tool_result",
                                **action,
                            }))

                        await websocket.send_text(json.dumps({"type": "done"}))

                    except Exception as e:
                        logger.exception("Pipeline error: %s", e)
                        audio_buffer.clear()
                        await websocket.send_text(
                            json.dumps({"type": "error", "message": "Processing failed. Please try again."})
                        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "message": "Connection error."})
            )
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8765")),
        reload=False,
    )

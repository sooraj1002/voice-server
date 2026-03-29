"""
Core voice pipeline: audio → STT → LLM (with tools) → TTS → audio.
"""
import logging
import subprocess

from typing import Awaitable, Callable

import stt
import tts
import llm
from llm import build_system_prompt
from tools import TOOL_DEFINITIONS, execute_tool, prepare_add_expense, execute_prepared_expense
from model_manager import model_manager

logger = logging.getLogger(__name__)


def _decode_webm_to_pcm(audio_bytes: bytes) -> bytes:
    """
    Decode WebM/Opus bytes to 16kHz mono 16-bit PCM using ffmpeg.
    """
    proc = subprocess.run(
        [
            "ffmpeg",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        logger.error("ffmpeg error: %s", proc.stderr.decode(errors="replace"))
        return b""
    return proc.stdout


async def process_turn(
    audio_bytes: bytes,
    jwt: str,
    conversation_history: list[dict],
    session_cache: dict,
    confirm_callback: Callable[[dict], Awaitable[bool]] | None = None,
) -> tuple[str, bytes, dict | None]:
    """
    Process one voice turn.

    confirm_callback: async (display: dict) -> bool
        Called before executing add_expense. Return True to confirm, False to cancel.
        If None, expenses are added without confirmation.

    Returns:
        (transcript, wav_bytes, action_event_or_None)
    """
    # 1. Decode audio
    pcm_bytes = _decode_webm_to_pcm(audio_bytes)
    if not pcm_bytes:
        return "", b"", None

    # 2. STT
    transcript = stt.transcribe(pcm_bytes)
    if not transcript or len(transcript) < 3:
        logger.debug("Empty/short transcript, skipping.")
        return transcript, b"", None

    logger.info("Transcript: %s", transcript)

    # 3. Add user message to history
    conversation_history.append({"role": "user", "content": transcript})

    # 4. First LLM call (may produce tool calls)
    result = await llm.chat_with_tools(conversation_history, TOOL_DEFINITIONS)

    last_action = None
    response_text = result["text"]

    # 5. Execute tool calls if any
    if result["tool_calls"]:
        # Append the assistant's tool-calling message
        conversation_history.append(
            {
                "role": "assistant",
                "content": result["text"],
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in result["tool_calls"]
                ],
            }
        )

        for tc in result["tool_calls"]:
            if tc["name"] == "add_expense" and confirm_callback is not None:
                # Resolve the expense details first, then ask the user to confirm
                display, payload, error = await prepare_add_expense(
                    tc["arguments"], jwt, session_cache
                )
                if error:
                    tool_result, action = error, None
                else:
                    confirmed = await confirm_callback(display)
                    if confirmed:
                        tool_result, action = await execute_prepared_expense(payload, jwt)
                    else:
                        tool_result, action = "User reviewed and cancelled the expense.", None
            else:
                tool_result, action = await execute_tool(
                    tc["name"], tc["arguments"], jwt, session_cache
                )

            if action:
                last_action = action
            logger.info("Tool %s → %s", tc["name"], tool_result)
            conversation_history.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tc["id"],
                }
            )

        # 6. Second LLM call to get final spoken response
        follow_up = await llm.chat_with_tools(conversation_history, TOOL_DEFINITIONS)
        response_text = follow_up["text"]

    if not response_text:
        response_text = "Done."

    # 7. Append assistant response to history
    conversation_history.append({"role": "assistant", "content": response_text})

    # 8. TTS
    wav_bytes = tts.synthesize(response_text)

    # 9. Touch model activity timer
    model_manager.touch()

    return transcript, wav_bytes, last_action

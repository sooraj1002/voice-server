"""
Ollama LLM client with tool calling support (qwen3:8b).
"""
import json
import logging
import os
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("VOICE_MODEL", "qwen3:8b")


def build_system_prompt(user_name: str = "") -> str:
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
    name_part = f" The user's name is {user_name}." if user_name else ""
    return (
        f"You are a voice assistant for a personal expense tracker.{name_part} "
        f"Today is {today}. "
        "Be concise — respond in 1-2 sentences unless the user asks for a list. "
        "Always confirm after adding an expense. "
        "If the amount seems unusually large (over 50000), confirm with the user before adding. "
        "When listing items, keep it brief. "
        "Do not use markdown formatting — your responses will be spoken aloud."
    )


async def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
) -> dict:
    """
    Send messages to Ollama and return {"text": str, "tool_calls": list}.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": 0.3},
            },
        )
        resp.raise_for_status()
        data = resp.json()

    message = data.get("message", {})
    content = message.get("content", "")
    tool_calls_raw = message.get("tool_calls") or []

    tool_calls = []
    for tc in tool_calls_raw:
        fn = tc.get("function", {})
        args = fn.get("arguments", {})
        # Ollama may return args as a JSON string
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        tool_calls.append(
            {
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": args,
            }
        )

    return {"text": content, "tool_calls": tool_calls}

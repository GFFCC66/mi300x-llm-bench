"""Parse Server-Sent Events from an OpenAI-compatible chat stream."""

from __future__ import annotations

import json
from typing import Any

DONE = object()


def parse_sse_line(line: str) -> Any:
    """Return a JSON object, ``DONE``, or ``None`` when the line carries nothing.

    A bare JSON object is accepted so a non-streaming response can be read by
    the same path. Callers must not treat that as a timed token stream.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if stripped.startswith("data:"):
        data = stripped[5:].strip()
        if data == "[DONE]":
            return DONE
        return json.loads(data)
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)
    return None


def _coerce_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def extract_delta_text(event: object) -> str:
    """Text from a streaming delta, or from a non-streaming message."""
    if not isinstance(event, dict):
        return ""
    choices = event.get("choices") or []
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    choice = choices[0]
    delta = choice.get("delta")
    if isinstance(delta, dict):
        text = _coerce_text(delta.get("content"))
        if text:
            return text
    message = choice.get("message")
    if isinstance(message, dict):
        text = _coerce_text(message.get("content"))
        if text:
            return text
    if isinstance(choice.get("text"), str):
        return choice["text"]
    return ""


def extract_usage(event: object) -> tuple[int | None, int | None] | None:
    """Return ``(prompt_tokens, completion_tokens)`` when a usage object exists."""
    if not isinstance(event, dict):
        return None
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    prompt_tokens = prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None
    completion_tokens = (
        completion if isinstance(completion, int) and not isinstance(completion, bool) else None
    )
    if prompt_tokens is None and completion_tokens is None:
        return None
    return prompt_tokens, completion_tokens

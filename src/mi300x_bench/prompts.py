"""Synthetic prompts and local prompt files.

``input_len`` is a target word count. It is a stand-in for tokens: these
words are often close to one token each on BPE tokenizers, but the count is
not exact for Llama or any other specific vocabulary. When the server returns
``usage.prompt_tokens``, that value is the one recorded in the summary.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Short, whitespace-separated words. The first word of a prompt is a fixed
# label so every request is obviously a benchmark prompt.
_WORDS = (
    "alpha",
    "bravo",
    "cedar",
    "delta",
    "ember",
    "frost",
    "grove",
    "hazel",
    "ivory",
    "juniper",
    "kelp",
    "lunar",
    "maple",
    "north",
    "opal",
    "pebble",
    "quartz",
    "river",
    "spruce",
    "timber",
    "umber",
    "violet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
)


def synthetic_prompt(input_len: int, index: int, seed: int) -> str:
    """Build a deterministic prompt of exactly ``input_len`` words."""
    if input_len < 1:
        raise ValueError("input_len must be >= 1")
    rng = random.Random((seed + index * 1_000_003) & 0xFFFFFFFF)
    words = [_WORDS[rng.randrange(len(_WORDS))] for _ in range(input_len)]
    words[0] = "benchmark"
    if input_len > 1:
        words[1] = f"id{index}"
    return " ".join(words)


def word_count(text: str) -> int:
    return len(text.split())


def _user_text_from_messages(messages: object) -> str:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from")
        content = message.get("content")
        if content is None:
            content = message.get("value")
        if role not in {None, "user", "human"}:
            continue
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    if not parts:
        raise ValueError("no user text in messages")
    return "\n".join(parts)


def prompt_from_record(record: object) -> str:
    """Accept a string, an OpenAI-style object, or a ShareGPT-style object."""
    if isinstance(record, str):
        text = record.strip()
        if not text:
            raise ValueError("empty prompt string")
        return text
    if not isinstance(record, dict):
        raise ValueError("prompt record must be a string or object")
    if isinstance(record.get("prompt"), str):
        text = record["prompt"].strip()
        if not text:
            raise ValueError("empty prompt field")
        return text
    if "messages" in record:
        return _user_text_from_messages(record["messages"])
    if "conversations" in record:
        return _user_text_from_messages(record["conversations"])
    raise ValueError("prompt record needs prompt, messages, or conversations")


def load_prompts(path: Path) -> list[str]:
    """Load prompts from JSONL or from a text file with one prompt per line."""
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    raw = path.read_text(encoding="utf-8")
    prompts: list[str] = []
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            try:
                prompts.append(prompt_from_record(record))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    else:
        for line in raw.splitlines():
            if line.strip():
                prompts.append(line.strip())
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


def prompts_for_requests(
    *,
    num_requests: int,
    input_len: int | None,
    seed: int,
    dataset: Path | None,
) -> list[str]:
    if num_requests < 1:
        raise ValueError("num_requests must be >= 1")
    if dataset is not None:
        loaded = load_prompts(dataset)
        return [loaded[i % len(loaded)] for i in range(num_requests)]
    if input_len is None:
        raise ValueError("input_len is required for synthetic prompts")
    return [synthetic_prompt(input_len, i, seed) for i in range(num_requests)]

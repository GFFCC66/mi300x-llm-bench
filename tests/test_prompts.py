from __future__ import annotations

import json
from pathlib import Path

import pytest

from mi300x_bench.prompts import load_prompts, prompt_from_record, synthetic_prompt, word_count
from mi300x_bench.sse import DONE, extract_delta_text, extract_usage, parse_sse_line


def test_synthetic_prompt_is_deterministic_and_exact() -> None:
    first = synthetic_prompt(32, 3, seed=7)
    assert word_count(first) == 32
    assert first.split()[0] == "benchmark"
    assert first.split()[1] == "id3"
    assert synthetic_prompt(32, 3, seed=7) == first
    assert synthetic_prompt(32, 4, seed=7) != first
    assert word_count(synthetic_prompt(1, 0, seed=0)) == 1
    assert word_count(synthetic_prompt(8192, 0, seed=1)) == 8192


def test_synthetic_prompt_rejects_zero() -> None:
    with pytest.raises(ValueError):
        synthetic_prompt(0, 0, 0)


def test_load_jsonl_and_text(tmp_path: Path) -> None:
    jsonl = tmp_path / "prompts.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "alpha beta"}),
                json.dumps({"messages": [{"role": "user", "content": "one two three"}]}),
                json.dumps(
                    {
                        "conversations": [
                            {"from": "human", "value": "sharegpt question"},
                            {"from": "gpt", "value": "ignored"},
                        ]
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_prompts(jsonl)
    assert loaded == ["alpha beta", "one two three", "sharegpt question"]

    text = tmp_path / "prompts.txt"
    text.write_text("line one\n\nline two\n", encoding="utf-8")
    assert load_prompts(text) == ["line one", "line two"]


def test_prompt_record_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prompt_from_record({"note": "missing"})
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        load_prompts(bad)


def test_sse_ignores_role_chunk_and_reads_usage() -> None:
    assert parse_sse_line("") is None
    assert parse_sse_line(": keep-alive") is None
    assert parse_sse_line("data: [DONE]") is DONE
    role = parse_sse_line('data: {"choices":[{"delta":{"role":"assistant"}}],"usage":null}')
    assert extract_delta_text(role) == ""
    content = parse_sse_line('data: {"choices":[{"delta":{"content":"Hi"}}]}')
    assert extract_delta_text(content) == "Hi"
    listed = parse_sse_line(
        'data: {"choices":[{"delta":{"content":[{"type":"text","text":"ab"}]}}]}'
    )
    assert extract_delta_text(listed) == "ab"
    usage_event = parse_sse_line(
        'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":2}}'
    )
    assert extract_usage(usage_event) == (4, 2)
    message = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    assert extract_delta_text(message) == "done"
    with pytest.raises(ValueError):
        parse_sse_line("data: {not-json")

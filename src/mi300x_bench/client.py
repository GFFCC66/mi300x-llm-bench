"""HTTP client for a streaming OpenAI-compatible chat completion."""

from __future__ import annotations

import time

import httpx

from mi300x_bench.metrics import RequestRecord
from mi300x_bench.sse import DONE, extract_delta_text, extract_usage, parse_sse_line


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    suffix = "/chat/completions"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base + "/models"


def build_payload(
    *,
    model: str,
    prompt: str,
    output_len: int,
    temperature: float,
    ignore_eos: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_len,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # vLLM and SGLang accept this extra field. It is omitted by default so a
    # strict OpenAI server is not sent an unknown parameter.
    if ignore_eos:
        payload["ignore_eos"] = True
    return payload


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _failure(
    *,
    start: float,
    end: float,
    error: str,
) -> RequestRecord:
    return RequestRecord(
        ok=False,
        error=error,
        start_abs=start,
        end_abs=end,
        ttft_s=None,
        itl_s=[],
        e2e_s=end - start,
        prompt_tokens=None,
        completion_tokens=None,
        chunk_count=0,
        streamed=False,
    )


async def check_server(client: httpx.AsyncClient, base_url: str) -> None:
    """GET ``/models``. Raises ``ConnectionError`` with a short message."""
    url = models_url(base_url)
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        raise ConnectionError(f"server not reachable at {url}: {exc}") from exc
    if response.status_code >= 400:
        body = response.text[:300]
        raise ConnectionError(f"GET {url} returned {response.status_code}: {body}")


async def chat_completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    prompt: str,
    output_len: int,
    temperature: float,
    ignore_eos: bool,
    api_key: str | None,
) -> RequestRecord:
    """Issue one streaming chat completion and time the content chunks."""
    payload = build_payload(
        model=model,
        prompt=prompt,
        output_len=output_len,
        temperature=temperature,
        ignore_eos=ignore_eos,
    )
    start = time.perf_counter()
    token_times: list[float] = []
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    streamed = False
    try:
        async with client.stream(
            "POST",
            chat_completions_url(base_url),
            json=payload,
            headers=_headers(api_key),
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", "replace")[:500]
                end = time.perf_counter()
                return _failure(start=start, end=end, error=f"HTTP {response.status_code}: {body}")
            async for line in response.aiter_lines():
                if line.strip().startswith("data:"):
                    streamed = True
                try:
                    event = parse_sse_line(line)
                except ValueError as exc:
                    end = time.perf_counter()
                    return _failure(start=start, end=end, error=f"invalid JSON in stream: {exc}")
                if event is None:
                    continue
                if event is DONE:
                    break
                usage = extract_usage(event)
                if usage is not None:
                    if usage[0] is not None:
                        prompt_tokens = usage[0]
                    if usage[1] is not None:
                        completion_tokens = usage[1]
                text = extract_delta_text(event)
                if text and streamed:
                    token_times.append(time.perf_counter())
        end = time.perf_counter()
    except httpx.HTTPError as exc:
        end = time.perf_counter()
        return _failure(start=start, end=end, error=f"connection failed: {exc}")

    ttft = (token_times[0] - start) if token_times else None
    itls = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
    if completion_tokens is None and token_times:
        completion_tokens = len(token_times)
    return RequestRecord(
        ok=True,
        error=None,
        start_abs=start,
        end_abs=end,
        ttft_s=ttft,
        itl_s=itls,
        e2e_s=end - start,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        chunk_count=len(token_times),
        streamed=streamed,
    )

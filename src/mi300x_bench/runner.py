"""Closed-loop concurrent benchmark runner."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx

from mi300x_bench import __version__
from mi300x_bench.client import chat_completion, check_server
from mi300x_bench.config import RunSpec
from mi300x_bench.metrics import RequestRecord, summarize
from mi300x_bench.prompts import prompts_for_requests


def _client(timeout_s: float) -> httpx.AsyncClient:
    # Ignore HTTP(S)_PROXY so the URL on the command line is the one we hit.
    timeout = httpx.Timeout(timeout_s)
    return httpx.AsyncClient(timeout=timeout, trust_env=False)


async def _bounded(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    spec: RunSpec,
    prompt: str,
) -> RequestRecord:
    async with semaphore:
        return await chat_completion(
            client,
            base_url=spec.base_url,
            model=spec.model,
            prompt=prompt,
            output_len=spec.output_len,
            temperature=spec.temperature,
            ignore_eos=spec.ignore_eos,
            api_key=spec.api_key,
        )


async def execute(spec: RunSpec) -> dict[str, object]:
    dataset = Path(spec.dataset) if spec.dataset else None
    prompts = prompts_for_requests(
        num_requests=spec.num_requests,
        input_len=spec.input_len,
        seed=spec.seed,
        dataset=dataset,
    )
    semaphore = asyncio.Semaphore(spec.concurrency)
    async with _client(spec.timeout_s) as client:
        if not spec.skip_health_check:
            await check_server(client, spec.base_url)
        if spec.warmup:
            warmup_prompts = prompts_for_requests(
                num_requests=spec.warmup,
                input_len=spec.input_len,
                seed=spec.seed + 1_000_000,
                dataset=dataset,
            )
            for prompt in warmup_prompts:
                warmup = await _bounded(semaphore, client, spec, prompt)
                if not warmup.ok:
                    raise ConnectionError(f"warmup request failed: {warmup.error}")
        tasks = [
            asyncio.create_task(_bounded(semaphore, client, spec, prompt)) for prompt in prompts
        ]
        records = list(await asyncio.gather(*tasks))
    return assemble_result(spec, records)


def assemble_result(spec: RunSpec, records: list[RequestRecord]) -> dict[str, object]:
    result: dict[str, object] = {
        "meta": {
            "benchmark": "mi300x_bench",
            "version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": spec.base_url,
            "model": spec.model,
            "concurrency": spec.concurrency,
            "target_input_len": spec.input_len,
            "output_len": spec.output_len,
            "num_requests": spec.num_requests,
            "warmup": spec.warmup,
            "temperature": spec.temperature,
            "ignore_eos": spec.ignore_eos,
            "seed": spec.seed,
            "dataset": spec.dataset,
            "prompt_source": spec.prompt_source,
            "methodology": (
                "closed-loop: at most `concurrency` requests are in flight; "
                "latencies start when each request is issued, after it acquires "
                "a concurrency slot; warmup requests are discarded"
            ),
        },
        "summary": summarize(records),
    }
    if spec.save_requests:
        result["requests"] = [_record_dict(record) for record in records]
    return result


def _record_dict(record: RequestRecord) -> dict[str, object]:
    return {
        "ok": record.ok,
        "error": record.error,
        "ttft_s": record.ttft_s,
        "e2e_s": record.e2e_s,
        "itl_s": record.itl_s,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "chunk_count": record.chunk_count,
        "streamed": record.streamed,
    }


def run_spec(spec: RunSpec) -> dict[str, object]:
    return asyncio.run(execute(spec))

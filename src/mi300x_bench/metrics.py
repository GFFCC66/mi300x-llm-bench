"""Latency and throughput summaries for one closed-loop benchmark point.

Latencies are client-observed:

- TTFT is the time from issuing the HTTP request to the first non-empty
  streamed content chunk. It does not include time spent waiting for a
  concurrency slot.
- ITL is the gap between consecutive non-empty streamed content chunks,
  pooled across requests. A chunk is often one token on vLLM, but servers
  may coalesce tokens, so ITL is a chunk gap, not a guaranteed per-token gap.
- TPOT is ``(e2e - ttft) / (completion_tokens - 1)`` for requests with at
  least two completion tokens. It uses the server usage count when present.
- E2E is the time from issuing the request until the response finishes.
- Throughput divides token and request counts by the wall clock from the
  earliest issued request to the latest completion (successful or failed).
  Failed requests are excluded from latency percentiles and from token
  totals. Warmup requests are not passed in here.
"""

from __future__ import annotations

from dataclasses import dataclass

from mi300x_bench.stats import percentile


@dataclass
class RequestRecord:
    ok: bool
    error: str | None
    start_abs: float
    end_abs: float
    ttft_s: float | None
    itl_s: list[float]
    e2e_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    chunk_count: int
    streamed: bool


def tpot_seconds(record: RequestRecord) -> float | None:
    """Average time per output token after the first, for one request."""
    if not record.ok or record.ttft_s is None:
        return None
    count = record.completion_tokens
    if count is None:
        count = record.chunk_count
    if count < 2:
        return None
    decode_s = record.e2e_s - record.ttft_s
    if decode_s < 0:
        return None
    return decode_s / (count - 1)


def _distribution(values: list[float], *, scale: float) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p99": None, "min": None, "max": None}
    return {
        "mean": scale * (sum(values) / len(values)),
        "p50": scale * percentile(values, 50),
        "p90": scale * percentile(values, 90),
        "p99": scale * percentile(values, 99),
        "min": scale * min(values),
        "max": scale * max(values),
    }


def _token_total(records: list[RequestRecord], field: str) -> int | None:
    values = [getattr(record, field) for record in records]
    if any(value is None for value in values):
        return None
    return int(sum(values))


def summarize(records: list[RequestRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("no request records to summarize")

    ok = [record for record in records if record.ok]
    failed = [record for record in records if not record.ok]
    duration_s = max(record.end_abs for record in records) - min(
        record.start_abs for record in records
    )
    if duration_s < 0:
        duration_s = 0.0

    prompt_total = _token_total(ok, "prompt_tokens")
    completion_total = _token_total(ok, "completion_tokens")

    def rate(count: int | None) -> float | None:
        if count is None or duration_s <= 0:
            return None
        return count / duration_s

    ttfts = [record.ttft_s for record in ok if record.ttft_s is not None]
    itls = [gap for record in ok for gap in record.itl_s]
    tpots = [value for record in ok if (value := tpot_seconds(record)) is not None]
    e2es = [record.e2e_s for record in ok]

    return {
        "duration_s": duration_s,
        "completed": len(ok),
        "failed": len(failed),
        "requests_per_s": rate(len(ok)),
        "output_tokens_per_s": rate(completion_total),
        "total_tokens_per_s": rate(
            None
            if prompt_total is None or completion_total is None
            else prompt_total + completion_total
        ),
        "prompt_tokens_total": prompt_total,
        "completion_tokens_total": completion_total,
        "prompt_tokens_mean": None if prompt_total is None or not ok else prompt_total / len(ok),
        "completion_tokens_mean": (
            None if completion_total is None or not ok else completion_total / len(ok)
        ),
        "ttft_ms": _distribution([float(value) for value in ttfts], scale=1000.0),
        "itl_ms": _distribution(itls, scale=1000.0),
        "tpot_ms": _distribution(tpots, scale=1000.0),
        "e2e_ms": _distribution(e2es, scale=1000.0),
        "errors": [record.error for record in failed if record.error][:10],
    }

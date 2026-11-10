from __future__ import annotations

import pytest

from mi300x_bench.metrics import RequestRecord, summarize, tpot_seconds
from mi300x_bench.stats import percentile


def _record(
    *,
    ok: bool = True,
    error: str | None = None,
    start: float = 0.0,
    end: float = 1.0,
    ttft: float | None = 0.1,
    itl: list[float] | None = None,
    prompt_tokens: int | None = 10,
    completion_tokens: int | None = 2,
    chunk_count: int | None = None,
    streamed: bool = True,
) -> RequestRecord:
    gaps = [] if itl is None else itl
    chunks = len(gaps) + (1 if ttft is not None else 0) if chunk_count is None else chunk_count
    return RequestRecord(
        ok=ok,
        error=error,
        start_abs=start,
        end_abs=end,
        ttft_s=ttft,
        itl_s=gaps,
        e2e_s=end - start,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        chunk_count=chunks,
        streamed=streamed,
    )


def test_percentile_interpolates_like_numpy_linear() -> None:
    sample = [10.0, 20.0, 30.0, 40.0]
    assert percentile(sample, 0) == 10
    assert percentile(sample, 50) == 25
    assert percentile(sample, 90) == pytest.approx(37)
    assert percentile(sample, 99) == pytest.approx(39.7)
    assert percentile(sample, 100) == 40
    assert percentile([7.0], 99) == 7


def test_percentile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_summarize_closed_loop_math() -> None:
    records = [
        _record(
            start=100.0,
            end=100.2,
            ttft=0.1,
            itl=[0.02, 0.04],
            prompt_tokens=10,
            completion_tokens=3,
        ),
        _record(
            start=100.0,
            end=100.4,
            ttft=0.3,
            itl=[0.05],
            prompt_tokens=10,
            completion_tokens=2,
        ),
        _record(
            ok=False,
            error="HTTP 500: no",
            start=100.1,
            end=100.3,
            ttft=None,
            itl=[],
            prompt_tokens=None,
            completion_tokens=None,
        ),
    ]
    summary = summarize(records)
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["duration_s"] == pytest.approx(0.4)
    assert summary["requests_per_s"] == pytest.approx(5)
    assert summary["output_tokens_per_s"] == pytest.approx(12.5)
    assert summary["total_tokens_per_s"] == pytest.approx(25 / 0.4)
    assert summary["prompt_tokens_total"] == 20
    assert summary["completion_tokens_total"] == 5
    ttft = summary["ttft_ms"]
    assert isinstance(ttft, dict)
    assert ttft["mean"] == pytest.approx(200)
    itl = summary["itl_ms"]
    assert isinstance(itl, dict)
    assert itl["min"] == pytest.approx(20)
    assert itl["max"] == pytest.approx(50)
    tpot = summary["tpot_ms"]
    assert isinstance(tpot, dict)
    assert tpot["min"] == pytest.approx(50)
    assert tpot["max"] == pytest.approx(100)
    assert summary["errors"] == ["HTTP 500: no"]


def test_tpot_needs_two_tokens_and_a_first_token() -> None:
    single = _record(ttft=0.2, completion_tokens=1, itl=[])
    assert tpot_seconds(single) is None
    unseen = _record(ttft=None, completion_tokens=8, chunk_count=0)
    assert tpot_seconds(unseen) is None
    fallback = _record(start=0, end=0.5, ttft=0.1, completion_tokens=None, chunk_count=5)
    assert tpot_seconds(fallback) == pytest.approx(0.1)


def test_percentiles_are_ordered_on_a_wider_sample() -> None:
    records = [
        _record(start=0, end=0.2 + i * 0.05, ttft=0.01 * (i + 1), itl=[0.01 * (i + 1)])
        for i in range(8)
    ]
    summary = summarize(records)
    for key in ("ttft_ms", "itl_ms", "e2e_ms"):
        dist = summary[key]
        assert isinstance(dist, dict)
        assert dist["p50"] <= dist["p90"] <= dist["p99"]

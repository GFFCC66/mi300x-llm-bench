from __future__ import annotations

import json
from pathlib import Path

from mi300x_bench.cli import main
from mi300x_bench.plot import plot_results


def _row(concurrency: int, input_len: int, throughput: float, ttft: float) -> dict[str, object]:
    summary = {
        "completed": 4,
        "failed": 0,
        "duration_s": 1.0,
        "requests_per_s": 4.0,
        "output_tokens_per_s": throughput,
        "total_tokens_per_s": throughput + input_len,
        "prompt_tokens_total": input_len * 4,
        "completion_tokens_total": 16,
        "prompt_tokens_mean": input_len,
        "completion_tokens_mean": 4,
        "ttft_ms": {"mean": ttft, "p50": ttft, "p90": ttft + 1, "p99": ttft + 2},
        "itl_ms": {"mean": 5, "p50": 5, "p90": 6, "p99": 7},
        "tpot_ms": {"mean": 8, "p50": 8, "p90": 9, "p99": 10},
        "e2e_ms": {"mean": 20, "p50": 20, "p90": 30, "p99": 40},
        "errors": [],
    }
    return {
        "meta": {
            "model": "mock",
            "base_url": "http://127.0.0.1/v1",
            "prompt_source": "synthetic",
            "dataset": None,
            "concurrency": concurrency,
            "target_input_len": input_len,
            "output_len": 4,
            "num_requests": 4,
            "warmup": 0,
            "temperature": 0,
            "ignore_eos": False,
            "seed": 0,
        },
        "summary": summary,
    }


def test_plot_single_and_sweep(tmp_path: Path) -> None:
    single = tmp_path / "one.json"
    single.write_text(json.dumps(_row(1, 128, 50, 12)), encoding="utf-8")
    single_png = tmp_path / "one.png"
    plot_results(single, single_png)
    assert single_png.stat().st_size > 1000

    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        json.dumps(
            {
                "runs": [
                    _row(1, 64, 40, 10),
                    _row(4, 64, 80, 15),
                    _row(1, 256, 20, 30),
                    _row(4, 256, 35, 45),
                ]
            }
        ),
        encoding="utf-8",
    )
    sweep_png = tmp_path / "sweep.png"
    assert main(["plot", "--input", str(sweep), "--output", str(sweep_png)]) == 0
    assert sweep_png.stat().st_size > 1000


def test_plot_rejects_unknown_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["plot", "--input", str(bad), "--output", str(tmp_path / "x.png")]) == 2

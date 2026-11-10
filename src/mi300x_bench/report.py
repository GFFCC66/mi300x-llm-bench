"""JSON and CSV writers plus a short text summary."""

from __future__ import annotations

import csv
import json
from pathlib import Path

_DISTANCES = ("ttft_ms", "itl_ms", "tpot_ms", "e2e_ms")
_STATS = ("mean", "p50", "p90", "p99")

SUMMARY_COLUMNS = [
    "model",
    "base_url",
    "prompt_source",
    "dataset",
    "concurrency",
    "target_input_len",
    "output_len",
    "num_requests",
    "warmup",
    "temperature",
    "ignore_eos",
    "seed",
    "completed",
    "failed",
    "duration_s",
    "requests_per_s",
    "output_tokens_per_s",
    "total_tokens_per_s",
    "prompt_tokens_total",
    "completion_tokens_total",
    "prompt_tokens_mean",
    "completion_tokens_mean",
    *[f"{distance}_{stat}" for distance in _DISTANCES for stat in _STATS],
]


def summary_row(result: dict[str, object]) -> dict[str, object]:
    meta = result["meta"]
    summary = result["summary"]
    if not isinstance(meta, dict) or not isinstance(summary, dict):
        raise ValueError("result is missing meta or summary")
    row: dict[str, object] = {
        "model": meta.get("model"),
        "base_url": meta.get("base_url"),
        "prompt_source": meta.get("prompt_source"),
        "dataset": meta.get("dataset"),
        "concurrency": meta.get("concurrency"),
        "target_input_len": meta.get("target_input_len"),
        "output_len": meta.get("output_len"),
        "num_requests": meta.get("num_requests"),
        "warmup": meta.get("warmup"),
        "temperature": meta.get("temperature"),
        "ignore_eos": meta.get("ignore_eos"),
        "seed": meta.get("seed"),
        "completed": summary.get("completed"),
        "failed": summary.get("failed"),
        "duration_s": summary.get("duration_s"),
        "requests_per_s": summary.get("requests_per_s"),
        "output_tokens_per_s": summary.get("output_tokens_per_s"),
        "total_tokens_per_s": summary.get("total_tokens_per_s"),
        "prompt_tokens_total": summary.get("prompt_tokens_total"),
        "completion_tokens_total": summary.get("completion_tokens_total"),
        "prompt_tokens_mean": summary.get("prompt_tokens_mean"),
        "completion_tokens_mean": summary.get("completion_tokens_mean"),
    }
    for distance in _DISTANCES:
        values = summary.get(distance)
        if not isinstance(values, dict):
            values = {}
        for stat in _STATS:
            row[f"{distance}_{stat}"] = values.get(stat)
    return row


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for result in results:
            row = summary_row(result)
            writer.writerow({key: _csv_cell(row.get(key)) for key in SUMMARY_COLUMNS})


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    return value


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return str(value)


def format_summary(result: dict[str, object]) -> str:
    meta = result["meta"]
    summary = result["summary"]
    assert isinstance(meta, dict)
    assert isinstance(summary, dict)
    lines = [
        (
            f"model={meta['model']} concurrency={meta['concurrency']} "
            f"input_len={meta['target_input_len']} output_len={meta['output_len']} "
            f"source={meta['prompt_source']}"
        ),
        (
            f"completed {summary['completed']}/{meta['num_requests']}  "
            f"failed {summary['failed']}  duration {_fmt(summary['duration_s'])} s"
        ),
        (
            f"output {_fmt(summary['output_tokens_per_s'])} tok/s   "
            f"total {_fmt(summary['total_tokens_per_s'])} tok/s   "
            f"requests {_fmt(summary['requests_per_s'])} req/s"
        ),
    ]
    for label, key in (
        ("TTFT ms", "ttft_ms"),
        ("ITL  ms", "itl_ms"),
        ("TPOT ms", "tpot_ms"),
        ("E2E  ms", "e2e_ms"),
    ):
        dist = summary[key]
        assert isinstance(dist, dict)
        lines.append(
            f"{label}  mean {_fmt(dist['mean'])}  p50 {_fmt(dist['p50'])}  "
            f"p90 {_fmt(dist['p90'])}  p99 {_fmt(dist['p99'])}"
        )
    lines.append(
        "prompt tokens "
        f"total {_fmt(summary['prompt_tokens_total'], 0)} "
        f"mean {_fmt(summary['prompt_tokens_mean'])}   "
        "completion tokens "
        f"total {_fmt(summary['completion_tokens_total'], 0)} "
        f"mean {_fmt(summary['completion_tokens_mean'])}"
    )
    errors = summary.get("errors") or []
    if isinstance(errors, list):
        for error in errors[:3]:
            lines.append(f"error: {error}")
    return "\n".join(lines)


def load_result_rows(path: Path) -> list[dict[str, object]]:
    """Load summary rows from a run JSON, a sweep JSON, or a summary CSV."""
    if path.suffix.lower() == ".csv":
        return _rows_from_csv(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "runs" in payload:
        runs = payload["runs"]
        if not isinstance(runs, list):
            raise ValueError("sweep JSON 'runs' must be a list")
        return [summary_row(item) for item in runs]
    if isinstance(payload, dict) and "summary" in payload:
        return [summary_row(payload)]
    raise ValueError("unrecognized results file; expected a run JSON, sweep JSON, or summary CSV")


def _rows_from_csv(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for raw in reader:
            row: dict[str, object] = {}
            for key, value in raw.items():
                row[key] = _coerce_csv(key, value)
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


_INTEGER_COLUMNS = {
    "concurrency",
    "target_input_len",
    "output_len",
    "num_requests",
    "warmup",
    "seed",
    "completed",
    "failed",
    "prompt_tokens_total",
    "completion_tokens_total",
}


def _coerce_csv(key: str, value: str | None) -> object:
    if value is None or value == "":
        return None
    if key in {"model", "base_url", "prompt_source", "dataset"}:
        return value
    if key == "ignore_eos":
        return value == "True"
    try:
        if key in _INTEGER_COLUMNS:
            return int(float(value))
        return float(value)
    except ValueError:
        return value

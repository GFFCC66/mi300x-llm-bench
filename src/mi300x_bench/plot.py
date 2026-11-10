"""Plot summary JSON or CSV with matplotlib.

One row draws latency percentile bars. A missing percentile is drawn as zero
on that chart. Several rows draw output throughput against concurrency (one
line per input length) and TTFT p50 against input length (one line per
concurrency).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from mi300x_bench.report import load_result_rows


def plot_results(source: Path, output: Path) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install the plot extra: pip install 'mi300x-bench[plot]'"
        ) from exc
    matplotlib.use("Agg", force=False)
    rows = load_result_rows(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 1:
        _plot_single(rows[0], output)
    else:
        _plot_sweep(rows, output)


def _number(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _plot_single(row: dict[str, object], output: Path) -> None:
    labels = ["TTFT", "ITL", "TPOT", "E2E"]
    keys = ["ttft_ms", "itl_ms", "tpot_ms", "e2e_ms"]
    stats = ["p50", "p90", "p99"]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    positions = list(range(len(labels)))
    for offset, stat in enumerate(stats):
        values = [_number(row, f"{key}_{stat}") or 0.0 for key in keys]
        ax.bar([pos + (offset - 1) * width for pos in positions], values, width=width, label=stat)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("milliseconds")
    model = row.get("model", "")
    concurrency = row.get("concurrency", "")
    ax.set_title(f"{model}  concurrency={concurrency}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_sweep(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    by_input: dict[object, list[dict[str, object]]] = defaultdict(list)
    by_conc: dict[object, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_input[row.get("target_input_len")].append(row)
        by_conc[row.get("concurrency")].append(row)

    ax0 = axes[0]
    for input_len, group in by_input.items():
        points = sorted(
            ((_number(row, "concurrency"), _number(row, "output_tokens_per_s")) for row in group),
            key=lambda item: (item[0] is None, item[0] or 0),
        )
        xs = [item[0] for item in points if item[0] is not None and item[1] is not None]
        ys = [item[1] for item in points if item[0] is not None and item[1] is not None]
        if xs:
            ax0.plot(xs, ys, marker="o", label=f"input {input_len}")
    ax0.set_xlabel("concurrency")
    ax0.set_ylabel("output tokens / s")
    ax0.set_title("Output throughput")
    ax0.legend()

    ax1 = axes[1]
    for concurrency, group in by_conc.items():
        points = sorted(
            ((_number(row, "target_input_len"), _number(row, "ttft_ms_p50")) for row in group),
            key=lambda item: (item[0] is None, item[0] or 0),
        )
        xs = [item[0] for item in points if item[0] is not None and item[1] is not None]
        ys = [item[1] for item in points if item[0] is not None and item[1] is not None]
        if xs:
            ax1.plot(xs, ys, marker="o", label=f"concurrency {concurrency}")
    ax1.set_xlabel("target input words")
    ax1.set_ylabel("TTFT p50 (ms)")
    ax1.set_title("Time to first token")
    ax1.legend()

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

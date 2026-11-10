"""Command-line interface.

Examples::

    python -m mi300x_bench run --base-url http://127.0.0.1:8000/v1 --model mock
    python -m mi300x_bench sweep --config configs/mock_sweep.yaml
    python -m mi300x_bench plot --input results/mock.json --output results/mock.png
    python -m mi300x_bench gpu-info
    python -m mi300x_bench mock --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mi300x_bench import __version__
from mi300x_bench.config import RunSpec, SweepConfig, load_sweep_config, specs_from_mapping
from mi300x_bench.gpu_info import collect_gpu_info, format_gpu_info
from mi300x_bench.report import format_summary, write_json, write_summary_csv
from mi300x_bench.runner import run_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mi300x-bench",
        description=(
            "Closed-loop benchmark for OpenAI-compatible LLM servers. "
            "Aimed at AMD Instinct MI300X, and runnable against any compatible endpoint."
        ),
    )
    parser.add_argument("--version", action="version", version=f"mi300x-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="benchmark one configuration")
    run.add_argument("--base-url", required=True, help="server root, usually http://host:8000/v1")
    run.add_argument("--model", required=True)
    run.add_argument("--api-key", default=None)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--input-len", type=int, default=256, help="synthetic prompt length in words")
    run.add_argument("--output-len", type=int, default=64, help="max_tokens sent to the server")
    run.add_argument("--num-requests", type=int, default=8)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument(
        "--ignore-eos",
        action="store_true",
        help="send ignore_eos=true (vLLM and SGLang; not part of the strict OpenAI schema)",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--timeout", type=float, default=600, help="per-request timeout in seconds")
    run.add_argument("--dataset", default=None, help="JSONL or one-prompt-per-line text file")
    run.add_argument(
        "--output",
        default=None,
        help="JSON path; a CSV with the same stem is written too",
    )
    run.add_argument("--save-requests", action="store_true")
    run.add_argument("--skip-health-check", action="store_true")

    sweep = sub.add_parser("sweep", help="run every point in a YAML config")
    sweep.add_argument("--config", required=True, type=Path)
    sweep.add_argument("--output-dir", default=None, help="overrides output_dir from the YAML")

    plot = sub.add_parser("plot", help="plot a run JSON, sweep JSON, or summary CSV")
    plot.add_argument("--input", required=True, type=Path, dest="plot_input")
    plot.add_argument("--output", required=True, type=Path, dest="plot_output")

    gpu = sub.add_parser("gpu-info", help="print rocm-smi product and VRAM info")
    gpu.add_argument("--json", action="store_true", dest="as_json")

    mock = sub.add_parser("mock", help="serve a tiny streaming chat endpoint for dry-runs")
    mock.add_argument("--host", default="127.0.0.1")
    mock.add_argument("--port", type=int, default=8000)
    mock.add_argument("--ttft-ms", type=float, default=0.0)
    mock.add_argument("--itl-ms", type=float, default=0.0)
    return parser


def _spec_from_args(args: argparse.Namespace) -> RunSpec:
    data: dict[str, object] = {
        "base_url": args.base_url,
        "model": args.model,
        "api_key": args.api_key,
        "concurrency": args.concurrency,
        "output_len": args.output_len,
        "num_requests": args.num_requests,
        "warmup": args.warmup,
        "temperature": args.temperature,
        "ignore_eos": bool(args.ignore_eos),
        "seed": args.seed,
        "timeout_s": args.timeout,
        "dataset": args.dataset,
        "save_requests": bool(args.save_requests),
        "skip_health_check": bool(args.skip_health_check),
    }
    if not args.dataset:
        data["input_len"] = args.input_len
    config = specs_from_mapping(data)
    return config.specs[0]


def _write_point(directory: Path, name: str, result: dict[str, object]) -> None:
    write_json(directory / f"{name}.json", result)
    write_summary_csv(directory / f"{name}.csv", [result])


def _point_name(spec: RunSpec, index: int) -> str:
    input_part = "dataset" if spec.input_len is None else f"in{spec.input_len}"
    return f"{index:02d}_{input_part}_out{spec.output_len}_c{spec.concurrency}"


def _exit_for(results: list[dict[str, object]]) -> int:
    if not results:
        return 2
    for result in results:
        summary = result["summary"]
        if isinstance(summary, dict) and summary.get("completed", 0):
            return 0
    return 2


def _print_progress(index: int, total: int, spec: RunSpec) -> None:
    print(
        f"[{index}/{total}] input_len={spec.input_len} output_len={spec.output_len} "
        f"concurrency={spec.concurrency}",
        flush=True,
    )


def command_run(args: argparse.Namespace) -> int:
    spec = _spec_from_args(args)
    try:
        result = run_spec(spec)
    except (ConnectionError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_summary(result))
    if args.output:
        output = Path(args.output)
        _write_point(output.parent, output.stem, result)
        print(f"wrote {output.parent / (output.stem + '.json')}")
        print(f"wrote {output.parent / (output.stem + '.csv')}")
    return _exit_for([result])


def command_sweep(args: argparse.Namespace) -> int:
    try:
        loaded = load_sweep_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir) if args.output_dir else loaded.output_dir
    config = SweepConfig(output_dir=output_dir, specs=loaded.specs, source=loaded.source)
    results: list[dict[str, object]] = []
    total = len(config.specs)
    for index, spec in enumerate(config.specs, start=1):
        _print_progress(index, total, spec)
        try:
            result = run_spec(spec)
        except (ConnectionError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            if results:
                print("stopping the sweep; earlier points were kept", file=sys.stderr)
            break
        results.append(result)
        print(format_summary(result))
        print("")
        _write_sweep(config, results)
        _write_point(output_dir / "points", _point_name(spec, index), result)
    if results:
        print(f"wrote {output_dir / 'sweep.json'}")
        print(f"wrote {output_dir / 'sweep.csv'}")
    if len(results) != total:
        return 2
    return _exit_for(results)


def _write_sweep(config: SweepConfig, results: list[dict[str, object]]) -> None:
    payload = {
        "config": str(config.source) if config.source else None,
        "runs": results,
    }
    write_json(config.output_dir / "sweep.json", payload)
    write_summary_csv(config.output_dir / "sweep.csv", results)


def command_plot(args: argparse.Namespace) -> int:
    from mi300x_bench.plot import plot_results

    try:
        plot_results(args.plot_input, args.plot_output)
    except SystemExit as exc:
        print(f"error: {exc.code}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.plot_output}")
    return 0


def command_gpu_info(args: argparse.Namespace) -> int:
    info = collect_gpu_info()
    if args.as_json:
        json.dump(info, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_gpu_info(info))
    return 0


def command_mock(args: argparse.Namespace) -> int:
    from mi300x_bench.mock_server import serve_blocking

    serve_blocking(args.host, args.port, args.ttft_ms / 1000.0, args.itl_ms / 1000.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "run": command_run,
        "sweep": command_sweep,
        "plot": command_plot,
        "gpu-info": command_gpu_info,
        "mock": command_mock,
    }
    return handlers[args.command](args)
